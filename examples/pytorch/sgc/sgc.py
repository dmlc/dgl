"""
This code was modified from the GCN implementation in DGL examples.
Simplifying Graph Convolutional Networks
Paper: https://arxiv.org/abs/1902.07153
Code: https://github.com/Tiiiger/SGC
SGC implementation in DGL.
"""
import argparse
import math
import time

import dgl
import dgl.function as fn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import (
    CiteseerGraphDataset,
    CoraGraphDataset,
    PubmedGraphDataset,
    register_data_args,
)
from dgl.nn.pytorch.conv import SGConv


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)[mask]  # only compute the evaluation set
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    # load and preprocess dataset
    if args.dataset == "cora":
        data = CoraGraphDataset()
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset()
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset()
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    in_feats = features.shape[1]
    n_classes = data.num_classes
    n_edges = g.num_edges()
    print(
        """----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d"""
        % (
            n_edges,
            n_classes,
            train_mask.int().sum().item(),
            val_mask.int().sum().item(),
            test_mask.int().sum().item(),
        )
    )

    n_edges = g.num_edges()
    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    # create SGC model
    model = SGConv(in_feats, n_classes, k=2, cached=True, bias=args.bias)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g, features)  # only compute the train set
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, g, features, labels, val_mask)
        print(
            "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
            "ETputs(KTEPS) {:.2f}".format(
                epoch,
                np.mean(dur),
                loss.item(),
                acc,
                n_edges / np.mean(dur) / 1000,
            )
        )

    print()
    acc = evaluate(model, g, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGC")
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=0.2, help="learning rate")
    parser.add_argument(
        "--bias", action="store_true", default=False, help="flag to use bias"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=100, help="number of training epochs"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-6, help="Weight for L2 loss"
    )
    args = parser.parse_args()
    print(args)

    main(args)
