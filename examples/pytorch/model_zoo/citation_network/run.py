import argparse
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from conf import *
from models import *

import dgl
from dgl.data import load_data, register_data_args


def get_model_and_config(name):
    name = name.lower()
    if name == "gcn":
        return GCN, GCN_CONFIG
    elif name == "gat":
        return GAT, GAT_CONFIG
    elif name == "graphsage":
        return GraphSAGE, GRAPHSAGE_CONFIG
    elif name == "appnp":
        return APPNP, APPNP_CONFIG
    elif name == "tagcn":
        return TAGCN, TAGCN_CONFIG
    elif name == "agnn":
        return AGNN, AGNN_CONFIG
    elif name == "sgc":
        return SGC, SGC_CONFIG
    elif name == "gin":
        return GIN, GIN_CONFIG
    elif name == "chebnet":
        return ChebNet, CHEBNET_CONFIG


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    # load and preprocess dataset
    data = load_data(args)
    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.to(args.gpu)
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

    # graph preprocess and calculate normalization factor
    # add self loop
    if args.self_loop:
        g = g.remove_self_loop().add_self_loop()
    n_edges = g.num_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata["norm"] = norm.unsqueeze(1)

    # create GCN model
    GNN, config = get_model_and_config(args.model)
    model = GNN(g, in_feats, n_classes, *config["extra_args"])

    if cuda:
        model = model.cuda()

    print(model)

    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    # initialize graph
    mean = 0
    for epoch in range(200):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            mean = (mean * (epoch - 3) + (time.time() - t0)) / (epoch - 2)
            acc = evaluate(model, features, labels, val_mask)
            print(
                "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                "ETputs(KTEPS) {:.2f}".format(
                    epoch,
                    mean,
                    loss.item(),
                    acc,
                    n_edges / mean / 1000,
                )
            )

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Node classification on citation networks."
    )
    register_data_args(parser)
    parser.add_argument(
        "--model",
        type=str,
        default="gcn",
        help="model to use, available models are gcn, gat, graphsage, gin,"
        "appnp, tagcn, sgc, agnn",
    )
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument(
        "--self-loop",
        action="store_true",
        help="graph self-loop (default=False)",
    )
    args = parser.parse_args()
    print(args)
    main(args)
