import argparse
import time
import traceback
from functools import partial

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
from ogb.nodeproppred import DglNodePropPredDataset
from sampler import ClusterIter, subgraph_collate_fn
from torch.utils.data import DataLoader

#### Neighbor sampler


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

    def forward(self, g, x):
        h = x
        for l, conv in enumerate(self.layers):
            h = conv(g, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
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
        h = x
        for l, conv in enumerate(self.layers):
            h = conv(g, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)

        return h


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, labels, val_nid, test_nid, batch_size, device):
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
        inputs = g.ndata["feat"]
        model = model.cpu()
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return (
        compute_acc(pred[val_nid], labels[val_nid]),
        compute_acc(pred[test_nid], labels[test_nid]),
        pred,
    )


def load_subtensor(g, labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata["feat"][input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


#### Entry point
def run(args, device, data):
    # Unpack data
    (
        train_nid,
        val_nid,
        test_nid,
        in_feats,
        labels,
        n_classes,
        g,
        cluster_iterator,
    ) = data

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
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    avg = 0
    iter_tput = []
    best_eval_acc = 0
    best_test_acc = 0
    for epoch in range(args.num_epochs):
        iter_load = 0
        iter_far = 0
        iter_back = 0
        iter_tl = 0
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_start = time.time()
        for step, cluster in enumerate(cluster_iterator):
            cluster = cluster.int().to(device)
            mask = cluster.ndata["train_mask"].to(device)
            if mask.sum() == 0:
                continue
            feat = cluster.ndata["feat"].to(device)
            batch_labels = cluster.ndata["labels"].to(device)
            tic_step = time.time()

            batch_pred = model(cluster, feat)
            batch_pred = batch_pred[mask]
            batch_labels = batch_labels[mask]
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            tic_far = time.time()
            loss.backward()
            optimizer.step()
            tic_back = time.time()
            iter_load += tic_step - tic_start
            iter_far += tic_far - tic_step
            iter_back += tic_back - tic_far

            tic_start = time.time()
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = (
                    th.cuda.max_memory_allocated() / 1000000
                    if th.cuda.is_available()
                    else 0
                )
                print(
                    "Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | GPU {:.1f} MB".format(
                        epoch, step, loss.item(), acc.item(), gpu_mem_alloc
                    )
                )

        toc = time.time()
        print(
            "Epoch Time(s): {:.4f} Load {:.4f} Forward {:.4f} Backward {:.4f}".format(
                toc - tic, iter_load, iter_far, iter_back
            )
        )
        if epoch >= 5:
            avg += toc - tic

        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc, test_acc, pred = evaluate(
                model, g, labels, val_nid, test_nid, args.val_batch_size, device
            )
            model = model.to(device)
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
    print("Avg epoch time: {}".format(avg / (epoch - 4)))
    return best_test_acc


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID. Use -1 for CPU training",
    )
    argparser.add_argument("--num-epochs", type=int, default=30)
    argparser.add_argument("--num-hidden", type=int, default=256)
    argparser.add_argument("--num-layers", type=int, default=3)
    argparser.add_argument("--batch-size", type=int, default=32)
    argparser.add_argument("--val-batch-size", type=int, default=10000)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--eval-every", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--save-pred", type=str, default="")
    argparser.add_argument("--wd", type=float, default=0)
    argparser.add_argument("--num_partitions", type=int, default=15000)
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device("cuda:%d" % args.gpu)
    else:
        device = th.device("cpu")

    # load ogbn-products data
    data = DglNodePropPredDataset(name="ogbn-products")
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    graph, labels = data[0]
    labels = labels[:, 0]
    num_nodes = train_idx.shape[0] + val_idx.shape[0] + test_idx.shape[0]
    assert num_nodes == graph.num_nodes()
    graph.ndata["labels"] = labels
    mask = th.zeros(num_nodes, dtype=th.bool)
    mask[train_idx] = True
    graph.ndata["train_mask"] = mask
    mask = th.zeros(num_nodes, dtype=th.bool)
    mask[val_idx] = True
    graph.ndata["valid_mask"] = mask
    mask = th.zeros(num_nodes, dtype=th.bool)
    mask[test_idx] = True
    graph.ndata["test_mask"] = mask

    graph.in_degrees(0)
    graph.out_degrees(0)
    graph.find_edges(0)

    cluster_iter_data = ClusterIter(
        "ogbn-products",
        graph,
        args.num_partitions,
        args.batch_size,
        th.cat([train_idx, val_idx, test_idx]),
    )
    idx = th.arange(args.num_partitions // args.batch_size)
    cluster_iterator = DataLoader(
        cluster_iter_data,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=partial(subgraph_collate_fn, graph),
    )

    in_feats = graph.ndata["feat"].shape[1]
    print(in_feats)
    n_classes = (labels.max() + 1).item()
    # Pack data
    data = (
        train_idx,
        val_idx,
        test_idx,
        in_feats,
        labels,
        n_classes,
        graph,
        cluster_iterator,
    )

    # Run 10 times
    test_accs = []
    for i in range(10):
        test_accs.append(run(args, device, data))
        print(
            "Average test accuracy:", np.mean(test_accs), "±", np.std(test_accs)
        )
