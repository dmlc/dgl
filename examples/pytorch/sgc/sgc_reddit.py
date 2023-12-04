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

import dgl.function as fn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import load_data, register_data_args
from dgl.nn.pytorch.conv import SGConv


def normalize(h):
    return (h - h.mean(0)) / h.std(0)


def evaluate(model, features, graph, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)[mask]  # only compute the evaluation set
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    # load and preprocess dataset
    args.dataset = "reddit-self-loop"
    data = load_data(args)
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
            g.ndata["train_mask"].int().sum().item(),
            g.ndata["val_mask"].int().sum().item(),
            g.ndata["test_mask"].int().sum().item(),
        )
    )

    # graph preprocess and calculate normalization factor
    n_edges = g.num_edges()
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata["norm"] = norm.unsqueeze(1)

    # create SGC model
    model = SGConv(
        in_feats, n_classes, k=2, cached=True, bias=True, norm=normalize
    )
    if args.gpu >= 0:
        model = model.cuda()

    # use optimizer
    optimizer = torch.optim.LBFGS(model.parameters())

    # define loss closure
    def closure():
        optimizer.zero_grad()
        output = model(g, features)[train_mask]
        loss_train = F.cross_entropy(output, labels[train_mask])
        loss_train.backward()
        return loss_train

    # initialize graph
    for epoch in range(args.n_epochs):
        model.train()
        optimizer.step(closure)

    acc = evaluate(model, features, g, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGC")
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument(
        "--bias", action="store_true", default=False, help="flag to use bias"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=2, help="number of training epochs"
    )
    args = parser.parse_args()
    print(args)

    main(args)
