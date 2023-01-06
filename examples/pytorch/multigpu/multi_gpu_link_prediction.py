import argparse
import os
import time

import dgl.function as fn

import dgl.nn as dglnn
import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as skm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.data import AsNodePredDataset, RedditDataset
from dgl.dataloading import (
    as_edge_prediction_sampler,
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from dgl.multiprocessing import shared_tensor
from ogb.nodeproppred import DglNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size, use_uva):
        g.ndata["h"] = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["h"])
        for l, layer in enumerate(self.layers):
            dataloader = DataLoader(
                g,
                torch.arange(g.num_nodes(), device=device),
                sampler,
                device=device,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                use_ddp=True,
                use_uva=use_uva,
            )
            # in order to prevent running out of GPU memory, allocate a
            # shared output tensor 'y' in host memory
            y = shared_tensor(
                (
                    g.num_nodes(),
                    self.hid_size
                    if l != len(self.layers) - 1
                    else self.out_size,
                )
            )
            for input_nodes, output_nodes, blocks in (
                tqdm.tqdm(dataloader) if dist.get_rank() == 0 else dataloader
            ):
                x = blocks[0].srcdata["h"]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # non_blocking (with pinned memory) to accelerate data transfer
                y[output_nodes] = h.to(y.device, non_blocking=True)
            # make sure all GPUs are done writing to 'y'
            dist.barrier()
            g.ndata["h"] = y if use_uva else y.to(device)

        g.ndata.pop("h")
        return y


class NegativeSampler(object):
    def __init__(self, g, k, neg_share=False, device=None):
        if device is None:
            device = g.device
        self.weights = g.in_degrees().float().to(device) ** 0.75
        self.k = k
        self.neg_share = neg_share

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        n = len(src)
        if self.neg_share and n % self.k == 0:
            dst = self.weights.multinomial(n, replacement=True)
            dst = dst.view(-1, 1, self.k).expand(-1, self.k, -1).flatten()
        else:
            dst = self.weights.multinomial(n * self.k, replacement=True)
        src = src.repeat_interleave(self.k)
        return src, dst


class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata["h"] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v("h", "h", "score"))
            pos_score = pos_graph.edata["score"]
        with neg_graph.local_scope():
            neg_graph.ndata["h"] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v("h", "h", "score"))
            neg_score = neg_graph.edata["score"]

        score = torch.cat([pos_score, neg_score])
        label = torch.cat(
            [torch.ones_like(pos_score), torch.zeros_like(neg_score)]
        ).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss


def compute_acc_unsupervised(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]
    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)
    lr = lm.LogisticRegression(multi_class="multinomial", max_iter=10000)
    lr.fit(emb[train_nids], train_labels)
    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average="micro")
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average="micro")
    return f1_micro_eval, f1_micro_test


def evaluate(proc_id, model, g, device, use_uva):
    model.eval()
    batch_size = 10000
    with torch.no_grad():
        pred = model.module.inference(g, device, batch_size, use_uva)
    return pred


def train(
    proc_id, nprocs, device, g, train_idx, val_idx, test_idx, model, use_uva
):
    # Create PyTorch DataLoader for constructing blocks
    n_edges = g.num_edges()
    train_seeds = torch.arange(n_edges).to(device)
    labels = g.ndata["label"].to("cpu")

    sampler = NeighborSampler([10, 25], prefetch_node_feats=["feat"])
    sampler = as_edge_prediction_sampler(
        sampler,
        exclude="reverse_id",
        # For each edge with ID e in Reddit dataset, the reverse edge is e ± |E|/2.
        reverse_eids=torch.cat(
            [torch.arange(n_edges // 2, n_edges), torch.arange(0, n_edges // 2)]
        ).to(train_seeds),
        # num_negs = 1, neg_share = False
        negative_sampler=NegativeSampler(
            g, 1, False, device if use_uva else None
        ),
    )
    train_dataloader = DataLoader(
        g,
        train_seeds,
        sampler,
        device=device,
        batch_size=10000,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_ddp=True,
        use_uva=use_uva,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fcn = CrossEntropyLoss()
    iter_pos = []
    iter_neg = []
    for epoch in range(10):
        tic = time.time()
        model.train()
        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(
            train_dataloader
        ):
            x = blocks[0].srcdata["feat"]
            y_hat = model(blocks, x)
            loss = loss_fcn(y_hat, pos_graph, neg_graph)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 20 == 0 and proc_id == 0:  # log every 20 steps
                # gpu memory reserved by PyTorch
                gpu_mem_alloc = (
                    torch.cuda.max_memory_allocated() / 1000000
                    if torch.cuda.is_available()
                    else 0
                )
                print(
                    f"Epoch {epoch:05d} | Step {step:05d} | Loss {loss.item():.4f} | GPU {gpu_mem_alloc:.1f} MB"
                )

        t = time.time() - tic
        if proc_id == 0:
            print(f"Epoch Time(s): {t:.4f}")
        if (epoch + 1) % 5 == 0:  # eval every 5 epochs
            pred = evaluate(proc_id, model, g, device, use_uva)  # in parallel
            if proc_id == 0:
                # only master proc does the accuracy computation
                eval_acc, test_acc = compute_acc_unsupervised(
                    pred, labels, train_idx, val_idx, test_idx
                )
                print(
                    f"Epoch {epoch:05d} | Eval F1-score {eval_acc:.4f} | Test F1-Score {test_acc:.4f}"
                )


def run(proc_id, nprocs, devices, g, data, mode):
    # find corresponding device for my rank
    device = devices[proc_id]
    torch.cuda.set_device(device)
    # initialize process group and unpack data for sub-processes
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12345",
        world_size=nprocs,
        rank=proc_id,
    )
    out_size, train_idx, val_idx, test_idx = data
    g = g.to(device if mode == "puregpu" else "cpu")
    # create GraphSAGE model (distributed)
    in_size = g.ndata["feat"].shape[1]
    model = SAGE(in_size, 16, 16).to(device)
    model = DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )
    # training + testing
    use_uva = mode == "mixed"
    train(
        proc_id, nprocs, device, g, train_idx, val_idx, test_idx, model, use_uva
    )
    # cleanup process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        choices=["ogbn-products", "reddit"],
        help="name of dataset (default: ogbn-products)",
    )
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["mixed", "puregpu"],
        help="Training mode. 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU(s) in use. Can be a list of gpu ids for multi-gpu training,"
        " e.g., 0,1,2,3.",
    )
    args = parser.parse_args()
    devices = list(map(int, args.gpu.split(",")))
    nprocs = len(devices)
    assert (
        torch.cuda.is_available()
    ), f"Must have GPUs to enable multi-gpu training."
    print(f"Training in {args.mode} mode using {nprocs} GPU(s)")

    # load and preprocess dataset
    print("Loading data")
    if args.dataset == "ogbn-products":
        # can it be AsLinkPredDataset?
        dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
    elif args.dataset == "reddit":
        dataset = AsNodePredDataset(RedditDataset(self_loop=False))

    g = dataset[0]
    # avoid creating certain graph formats in each sub-process to save momory
    g.create_formats_()
    # thread limiting to avoid resource competition
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // nprocs)
    data = (
        dataset.num_classes,
        dataset.train_idx,
        dataset.val_idx,
        dataset.test_idx,
    )

    mp.spawn(run, args=(nprocs, devices, g, data, args.mode), nprocs=nprocs)
