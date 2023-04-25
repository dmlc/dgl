import argparse
import math
import time
import traceback

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn

import numpy as np
import torch as th
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from dgl.data import RedditDataset
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


class SAGEConvWithCV(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super().__init__()
        self.W = nn.Linear(in_feats * 2, out_feats)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, block, H, HBar=None):
        if self.training:
            with block.local_scope():
                H_src, H_dst = H
                HBar_src, agg_HBar_dst = HBar
                block.dstdata["agg_hbar"] = agg_HBar_dst
                block.srcdata["hdelta"] = H_src - HBar_src
                block.update_all(
                    fn.copy_u("hdelta", "m"), fn.mean("m", "hdelta_new")
                )
                h_neigh = (
                    block.dstdata["agg_hbar"] + block.dstdata["hdelta_new"]
                )
                h = self.W(th.cat([H_dst, h_neigh], 1))
                if self.activation is not None:
                    h = self.activation(h)
                return h
        else:
            with block.local_scope():
                H_src, H_dst = H
                block.srcdata["h"] = H_src
                block.update_all(fn.copy_u("h", "m"), fn.mean("m", "h_new"))
                h_neigh = block.dstdata["h_new"]
                h = self.W(th.cat([H_dst, h_neigh], 1))
                if self.activation is not None:
                    h = self.activation(h)
                return h


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConvWithCV(in_feats, n_hidden, activation))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConvWithCV(n_hidden, n_hidden, activation))
        self.layers.append(SAGEConvWithCV(n_hidden, n_classes, None))

    def forward(self, blocks):
        h = blocks[0].srcdata["features"]
        updates = []
        for layer, block in zip(self.layers, blocks):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[: block.number_of_dst_nodes()]
            hbar_src = block.srcdata["hist"]
            agg_hbar_dst = block.dstdata["agg_hist"]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst), (hbar_src, agg_hbar_dst))
            block.dstdata["h_new"] = h
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = th.arange(g.num_nodes())
        for l, layer in enumerate(self.layers):
            y = g.ndata["hist_%d" % (l + 1)]

            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(
                    dgl.in_subgraph(g, batch_nodes), batch_nodes
                )
                induced_nodes = block.srcdata[dgl.NID]

                h = x[induced_nodes].to(device)
                block = block.to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))

                y[start:end] = h.cpu()

            x = y
        return y


class NeighborSampler(object):
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(seeds)
        blocks = []
        hist_blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout)
            # For history aggregation we sample all neighbors.
            hist_frontier = dgl.in_subgraph(self.g, seeds)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            hist_block = dgl.to_block(hist_frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
            hist_blocks.insert(0, hist_block)
        return blocks, hist_blocks


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, labels, val_mask, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        inputs = g.ndata["features"]
        pred = model.inference(
            g, inputs, batch_size, device
        )  # also recomputes history tensors
    model.train()
    return compute_acc(pred[val_mask], labels[val_mask])


def load_subtensor(
    g, labels, blocks, hist_blocks, dev_id, aggregation_on_device=False
):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    blocks[0].srcdata["features"] = g.ndata["features"][
        blocks[0].srcdata[dgl.NID]
    ]
    blocks[-1].dstdata["label"] = labels[blocks[-1].dstdata[dgl.NID]]
    ret_blocks = []
    ret_hist_blocks = []
    for i, (block, hist_block) in enumerate(zip(blocks, hist_blocks)):
        hist_col = "features" if i == 0 else "hist_%d" % i
        block.srcdata["hist"] = g.ndata[hist_col][block.srcdata[dgl.NID]]

        # Aggregate history
        hist_block.srcdata["hist"] = g.ndata[hist_col][
            hist_block.srcdata[dgl.NID]
        ]
        if aggregation_on_device:
            hist_block = hist_block.to(dev_id)
            hist_block.srcdata["hist"] = hist_block.srcdata["hist"]
        hist_block.update_all(fn.copy_u("hist", "m"), fn.mean("m", "agg_hist"))

        block = block.to(dev_id)
        if not aggregation_on_device:
            hist_block = hist_block.to(dev_id)
        block.dstdata["agg_hist"] = hist_block.dstdata["agg_hist"]
        ret_blocks.append(block)
        ret_hist_blocks.append(hist_block)
    return ret_blocks, ret_hist_blocks


def create_history_storage(g, args, n_classes):
    # Initialize history storage
    for l in range(args.num_layers):
        dim = args.num_hidden if l != args.num_layers - 1 else n_classes
        g.ndata["hist_%d" % (l + 1)] = th.zeros(
            g.num_nodes(), dim
        ).share_memory_()


def init_history(g, model, dev_id, batch_size):
    with th.no_grad():
        model.inference(
            g, g.ndata["features"], batch_size, dev_id
        )  # replaces hist_i features in-place


def update_history(g, blocks):
    with th.no_grad():
        for i, block in enumerate(blocks):
            ids = block.dstdata[dgl.NID].cpu()
            hist_col = "hist_%d" % (i + 1)

            h_new = block.dstdata["h_new"].cpu()
            g.ndata[hist_col][ids] = h_new


def run(proc_id, n_gpus, args, devices, data):
    dropout = 0.2

    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = "tcp://{master_ip}:{master_port}".format(
            master_ip="127.0.0.1", master_port="12345"
        )
        world_size = n_gpus
        th.distributed.init_process_group(
            backend="nccl",
            init_method=dist_init_method,
            world_size=world_size,
            rank=proc_id,
        )
    th.cuda.set_device(dev_id)

    # Unpack data
    train_mask, val_mask, in_feats, labels, n_classes, g = data
    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()

    # Create sampler
    sampler = NeighborSampler(g, [int(_) for _ in args.fan_out.split(",")])

    # Create PyTorch DataLoader for constructing blocks
    if n_gpus > 1:
        dist_sampler = th.utils.data.distributed.DistributedSampler(
            train_nid.numpy(), shuffle=True, drop_last=False
        )
        dataloader = DataLoader(
            dataset=train_nid.numpy(),
            batch_size=args.batch_size,
            collate_fn=sampler.sample_blocks,
            sampler=dist_sampler,
            num_workers=args.num_workers_per_gpu,
        )
    else:
        dataloader = DataLoader(
            dataset=train_nid.numpy(),
            batch_size=args.batch_size,
            collate_fn=sampler.sample_blocks,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers_per_gpu,
        )

    # Define model
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu)

    # Move the model to GPU and define optimizer
    model = model.to(dev_id)
    if n_gpus > 1:
        model = DistributedDataParallel(
            model, device_ids=[dev_id], output_device=dev_id
        )
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(dev_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Compute history tensor and their aggregation before training on CPU
    model.eval()
    if n_gpus > 1:
        if proc_id == 0:
            init_history(g, model.module, dev_id, args.val_batch_size)
        th.distributed.barrier()
    else:
        init_history(g, model, dev_id, args.val_batch_size)
    model.train()

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        if n_gpus > 1:
            dist_sampler.set_epoch(epoch)
        tic = time.time()
        model.train()
        for step, (blocks, hist_blocks) in enumerate(dataloader):
            if proc_id == 0:
                tic_step = time.time()

            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            seeds = blocks[-1].dstdata[dgl.NID]

            blocks, hist_blocks = load_subtensor(
                g, labels, blocks, hist_blocks, dev_id, True
            )

            # forward
            batch_pred = model(blocks)
            # update history
            update_history(g, blocks)
            # compute loss
            batch_labels = blocks[-1].dstdata["label"]
            loss = loss_fcn(batch_pred, batch_labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if proc_id == 0:
                iter_tput.append(len(seeds) * n_gpus / (time.time() - tic_step))
            if step % args.log_every == 0 and proc_id == 0:
                acc = compute_acc(batch_pred, batch_labels)
                print(
                    "Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f}".format(
                        epoch,
                        step,
                        loss.item(),
                        acc.item(),
                        np.mean(iter_tput[3:]),
                    )
                )

        if n_gpus > 1:
            th.distributed.barrier()

        toc = time.time()
        if proc_id == 0:
            print("Epoch Time(s): {:.4f}".format(toc - tic))
            if epoch >= 5:
                avg += toc - tic
            if epoch % args.eval_every == 0 and epoch != 0:
                model.eval()
                eval_acc = evaluate(
                    model if n_gpus == 1 else model.module,
                    g,
                    labels,
                    val_nid,
                    args.val_batch_size,
                    dev_id,
                )
                print("Eval Acc {:.4f}".format(eval_acc))

    if n_gpus > 1:
        th.distributed.barrier()
    if proc_id == 0:
        print("Avg epoch time: {}".format(avg / (epoch - 4)))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument("--gpu", type=str, default="0")
    argparser.add_argument("--num-epochs", type=int, default=20)
    argparser.add_argument("--num-hidden", type=int, default=16)
    argparser.add_argument("--num-layers", type=int, default=2)
    argparser.add_argument("--fan-out", type=str, default="1,1")
    argparser.add_argument("--batch-size", type=int, default=1000)
    argparser.add_argument("--val-batch-size", type=int, default=1000)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--eval-every", type=int, default=5)
    argparser.add_argument("--lr", type=float, default=0.003)
    argparser.add_argument("--num-workers-per-gpu", type=int, default=0)
    args = argparser.parse_args()

    devices = list(map(int, args.gpu.split(",")))
    n_gpus = len(devices)

    # load reddit data
    data = RedditDataset(self_loop=True)
    n_classes = data.num_classes
    g = data[0]
    features = g.ndata["feat"]
    in_feats = features.shape[1]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    g.ndata["features"] = features.share_memory_()
    create_history_storage(g, args, n_classes)

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    g.create_formats_()
    # Pack data
    data = train_mask, val_mask, in_feats, labels, n_classes, g

    if n_gpus == 1:
        run(0, n_gpus, args, devices, data)
    else:
        mp.spawn(run, args=(n_gpus, args, devices, data), nprocs=n_gpus)
