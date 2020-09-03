#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import random
import time

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from ogb.nodeproppred import DglNodePropPredDataset

import dgl.nn.pytorch as dglnn


class GCN(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            self.convs.append(dglnn.GraphConv(in_hidden, out_hidden, "both"))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_hidden))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        for l in range(self.n_layers):
            h = self.convs[l](graph, h)
            if l < self.n_layers - 1:
                h = self.bns[l](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def cross_entropy(x, labels):
    y = F.cross_entropy(x, labels, reduction="none")
    y = th.log(0.5 + y) - math.log(0.5)
    return th.mean(y)


def train(model, graph, labels, train_idx, optimizer):
    model.train()

    feat = graph.ndata["feat"]
    mask = th.rand(train_idx.shape) < 0.5
    # mask = th.ones(train_idx.shape, dtype=th.bool)

    optimizer.zero_grad()
    pred = model(graph, feat)
    loss = cross_entropy(pred[train_idx[mask]], labels[train_idx[mask]])
    loss.backward()
    optimizer.step()

    return loss, pred


@th.no_grad()
def evaluate(model, graph, labels, train_idx, val_idx, test_idx):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    graph : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_idx : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    """
    model.eval()

    feat = graph.ndata["feat"]
    pred = model(graph, feat)
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx]),
        compute_acc(pred[val_idx], labels[val_idx]),
        compute_acc(pred[test_idx], labels[test_idx]),
        val_loss,
    )


def warmup_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        lr *= epoch / 50

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def run(args, device, graph, in_feats, n_classes, labels, train_idx, val_idx, test_idx):
    # Define model and optimizer
    model = GCN(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout)

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=100, verbose=True)

    # Training loop
    total_time = 0
    best_val_acc = 0
    best_test_acc = 0
    best_val_loss = float("inf")

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        warmup_learning_rate(optimizer, args.lr, epoch)

        loss, pred = train(model, graph, labels, train_idx, optimizer)

        acc = compute_acc(pred[train_idx], labels[train_idx])

        toc = time.time()
        total_time += toc - tic

        train_acc, val_acc, test_acc, val_loss = evaluate(model, graph, labels, train_idx, val_idx, test_idx)

        lr_scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
            best_val_acc = val_acc.item()
            best_test_acc = test_acc.item()

        if epoch % args.log_every == 0:
            print(f"*** Epoch: {epoch} ***")
            print(
                f"Loss: {loss.item():.4f}, Acc: {acc.item():.4f}\n"
                f"Train/Val/Best val Acc: {train_acc:.4f}/{val_acc:.4f}/{best_val_acc:.4f}, Val Loss: {val_loss:.4f}, Test Acc: {best_test_acc:.4f}"
            )

    print("******")
    print(f"Avg epoch time: {total_time / args.n_epochs}")

    return best_val_acc, best_test_acc


def main():
    argparser = argparse.ArgumentParser("OGBN-Arxiv")
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--n-runs", type=int, default=10)
    argparser.add_argument("--n-epochs", type=int, default=1000)
    argparser.add_argument("--n-layers", type=int, default=3)
    argparser.add_argument("--n-hidden", type=int, default=256)
    argparser.add_argument("--lr", type=float, default=0.005)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--wd", type=float, default=0)
    argparser.add_argument("--log-every", type=int, default=20)
    args = argparser.parse_args()

    if args.cpu:
        device = th.device("cpu")
    else:
        device = th.device("cuda:%d" % args.gpu)

    # load data
    data = DglNodePropPredDataset(name="ogbn-arxiv")
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    labels = labels[:, 0]

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    in_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()
    graph.create_format_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.to(device)
    graph = graph.to(device)

    # run
    val_accs = []
    test_accs = []

    for i in range(args.n_runs):
        val_acc, test_acc = run(args, device, graph, in_feats, n_classes, labels, train_idx, val_idx, test_idx)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    print(f"Runned {args.n_runs} times")
    print("Val Accs:", val_accs)
    print("Test Accs:", test_accs)
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")


if __name__ == "__main__":
    main()

