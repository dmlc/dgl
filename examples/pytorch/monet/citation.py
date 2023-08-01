import argparse
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import DGLGraph
from dgl.data import load_data, register_data_args
from dgl.nn.pytorch.conv import GMMConv


class MoNet(nn.Module):
    def __init__(
        self,
        g,
        in_feats,
        n_hidden,
        out_feats,
        n_layers,
        dim,
        n_kernels,
        dropout,
    ):
        super(MoNet, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()

        # Input layer
        self.layers.append(GMMConv(in_feats, n_hidden, dim, n_kernels))
        self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        # Hidden layer
        for _ in range(n_layers - 1):
            self.layers.append(GMMConv(n_hidden, n_hidden, dim, n_kernels))
            self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        # Output layer
        self.layers.append(GMMConv(n_hidden, out_feats, dim, n_kernels))
        self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat, pseudo):
        h = feat
        for i in range(len(self.layers)):
            if i != 0:
                h = self.dropout(h)
            h = self.layers[i](self.g, h, self.pseudo_proj[i](pseudo))
        return h


def evaluate(model, features, pseudo, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features, pseudo)
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
            train_mask.sum().item(),
            val_mask.sum().item(),
            test_mask.sum().item(),
        )
    )

    # graph preprocess and calculate normalization factor
    g = g.remove_self_loop().add_self_loop()
    n_edges = g.num_edges()
    us, vs = g.edges(order="eid")
    udeg, vdeg = 1 / torch.sqrt(g.in_degrees(us).float()), 1 / torch.sqrt(
        g.in_degrees(vs).float()
    )
    pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)

    # create GraphSAGE model
    model = MoNet(
        g,
        in_feats,
        args.n_hidden,
        n_classes,
        args.n_layers,
        args.pseudo_dim,
        args.n_kernels,
        args.dropout,
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
        logits = model(features, pseudo)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            mean = (mean * (epoch - 3) + (time.time() - t0)) / (epoch - 2)
            acc = evaluate(model, features, pseudo, labels, val_mask)
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
    acc = evaluate(model, features, pseudo, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoNet on citation network")
    register_data_args(parser)
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout probability"
    )
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument(
        "--n-epochs", type=int, default=200, help="number of training epochs"
    )
    parser.add_argument(
        "--n-hidden", type=int, default=16, help="number of hidden gcn units"
    )
    parser.add_argument(
        "--n-layers", type=int, default=1, help="number of hidden gcn layers"
    )
    parser.add_argument(
        "--pseudo-dim",
        type=int,
        default=2,
        help="Pseudo coordinate dimensions in GMMConv, 2 for cora and 3 for pubmed",
    )
    parser.add_argument(
        "--n-kernels",
        type=int,
        default=3,
        help="Number of kernels in GMMConv layer",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight for L2 loss"
    )
    args = parser.parse_args()
    print(args)

    main(args)
