import argparse
import time

import dgl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from appnp import APPNP
from dgl.data import (
    CiteseerGraphDataset,
    CoraGraphDataset,
    PubmedGraphDataset,
    register_data_args,
)


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

    n_edges = g.num_edges()
    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    # create APPNP model
    model = APPNP(
        g,
        in_feats,
        args.hidden_sizes,
        n_classes,
        F.relu,
        args.in_drop,
        args.edge_drop,
        args.alpha,
        args.k,
    )

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # initialize graph
    mean = 0
    for epoch in range(args.n_epochs):
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
    parser = argparse.ArgumentParser(description="APPNP")
    register_data_args(parser)
    parser.add_argument(
        "--in-drop", type=float, default=0.5, help="input feature dropout"
    )
    parser.add_argument(
        "--edge-drop", type=float, default=0.5, help="edge propagation dropout"
    )
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument(
        "--n-epochs", type=int, default=200, help="number of training epochs"
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[64],
        help="hidden unit sizes for appnp",
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Number of propagation steps"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Teleport Probability"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight for L2 loss"
    )
    args = parser.parse_args()
    print(args)

    main(args)
