"""
This is a modified version of: https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/ogbn-products/graphsage/main.py
This example shows how to enable ARGO to automatically instantiate multi-processing and adjust CPU core assignment to achieve better performance.
"""

import argparse

import ctypes
import os
import time
from multiprocessing import RawValue

import dgl
import dgl.nn.pytorch as dglnn

import numpy as np
import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from argo import ARGO
from ogb.nodeproppred import DglNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel

avg_total = RawValue(ctypes.c_float, 0.0)


class SAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[: block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device):
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
        for l, layer in enumerate(self.layers):
            y = th.zeros(
                g.num_nodes(),
                self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
            ).to(device)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                th.arange(g.num_nodes()),
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(
                dataloader, disable=None
            ):
                block = blocks[0].int().to(device)

                h = x[input_nodes]
                h_dst = h[: block.num_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h

            x = y
        return y


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, val_nid, test_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.module.inference(g, nfeat, device)
    model.train()
    return (
        compute_acc(pred[val_nid], labels[val_nid]),
        compute_acc(pred[test_nid], labels[test_nid]),
        pred,
    )


def load_subtensor(nfeat, labels, seeds, input_nodes):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs = nfeat[input_nodes]
    batch_labels = labels[seeds]
    return batch_inputs, batch_labels


#### Entry point
def train(
    args,
    device,
    data,
    rank,
    world_size,
    comp_core,
    load_core,
    counter,
    b_size,
    ep,
):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )
    dataloader = dgl.dataloading.DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=b_size,
        shuffle=True,
        drop_last=False,
        num_workers=len(load_core),
        use_ddp=True,
    )

    # Define model and optimizer
    model = SAGE(
        in_feats,
        args.num_hidden,
        n_classes,
        args.num_layers,
        F.relu,
        args.dropout,
    )
    model = model.to(device)
    model = DistributedDataParallel(model)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    avg = 0
    iter_tput = []
    best_eval_acc = 0
    best_test_acc = 0
    PATH = "model.pt"
    if counter[0] != 0:
        checkpoint = th.load(PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]

    with dataloader.enable_cpu_affinity(
        loader_cores=load_core, compute_cores=comp_core
    ):
        for epoch in range(ep):
            tic = time.time()

            # Loop over the dataloader to sample the computation dependency graph as a list of
            # blocks.
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                tic_step = time.time()

                # copy block to gpu
                blocks = [blk.int().to(device) for blk in blocks]

                # Load the input features as well as output labels
                batch_inputs, batch_labels = load_subtensor(
                    nfeat, labels, seeds, input_nodes
                )

                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_tput.append(len(seeds) / (time.time() - tic_step))
                if step % args.log_every == 0 and step != 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    print(
                        "Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f}".format(
                            step,
                            loss.item(),
                            acc.item(),
                            np.mean(iter_tput[3:]),
                        )
                    )

            toc = time.time()
            print("Epoch Time(s): {:.4f}".format(toc - tic))

            if rank == 0:
                global avg_total
                avg_total.value += toc - tic
                avg += toc - tic

                if epoch % args.eval_every == 0 and epoch != 0:
                    eval_acc, test_acc, pred = evaluate(
                        model, g, nfeat, labels, val_nid, test_nid, device
                    )
                    if args.save_pred:
                        np.savetxt(
                            args.save_pred + "%02d" % epoch,
                            pred.argmax(1).cpu().numpy(),
                            "%d",
                        )
                    print("Eval Acc {:.4f}".format(eval_acc))
                    if eval_acc > best_eval_acc:
                        best_eval_acc = eval_acc
                        best_test_acc = test_acc
                    print(
                        "Best Eval Acc {:.4f} Test Acc {:.4f}".format(
                            best_eval_acc, best_test_acc
                        )
                    )

    dist.barrier()
    if rank == 0:
        th.save(
            {
                "epoch": counter[0],
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            PATH,
        )
        if args.num_epochs == counter[0] + epoch + 1:
            print(
                "Avg epoch time: {}".format(avg_total.value / args.num_epochs)
            )
            print(
                "Avg epoch time after auto-tuning: {}".format(avg / (epoch + 1))
            )

    return best_test_acc


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID. Use -1 for CPU training",
    )
    argparser.add_argument("--num-epochs", type=int, default=20)
    argparser.add_argument("--num-hidden", type=int, default=256)
    argparser.add_argument("--num-layers", type=int, default=3)
    argparser.add_argument("--fan-out", type=str, default="5,10,15")
    argparser.add_argument("--batch-size", type=int, default=1000)
    argparser.add_argument("--val-batch-size", type=int, default=10000)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--eval-every", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=0.003)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        choices=["ogbn-papers100M", "ogbn-products"],
    )
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of sampling processes. Use 0 for no extra process.",
    )
    argparser.add_argument("--save-pred", type=str, default="")
    argparser.add_argument("--wd", type=float, default=0)
    args = argparser.parse_args()

    device = th.device("cpu")

    # load ogbn-products data
    data = DglNodePropPredDataset(args.dataset)
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    graph, labels = data[0]
    nfeat = graph.ndata.pop("feat").to(device)
    labels = labels[:, 0].to(device)

    in_feats = nfeat.shape[1]
    n_classes = (labels.max() + 1).item()
    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    graph.create_formats_()
    # Pack data
    data = (
        train_idx,
        val_idx,
        test_idx,
        in_feats,
        labels,
        n_classes,
        nfeat,
        graph,
    )
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    mp.set_start_method("fork", force=True)
    runtime = ARGO(
        n_search=15, epoch=args.num_epochs, batch_size=args.batch_size
    )  # initialization
    runtime.run(train, args=(args, device, data))  # wrap the training function
