"""
Graph Attention Networks v2 (GATv2) in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
"""

import argparse
import time

import dgl

import numpy as np
import torch
import torch.nn.functional as F
from dgl.data import (
    CiteseerGraphDataset,
    CoraGraphDataset,
    PubmedGraphDataset,
    register_data_args,
)
from gatv2 import GATv2


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decrease."""
        torch.save(model.state_dict(), "es_checkpoint.pt")


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(g, model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


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
    num_feats = features.shape[1]
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

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.num_edges()
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GATv2(
        args.num_layers,
        num_feats,
        args.num_hidden,
        n_classes,
        heads,
        F.elu,
        args.in_drop,
        args.attn_drop,
        args.negative_slope,
        args.residual,
    )
    print(model)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # initialize graph
    mean = 0
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            mean = (mean * (epoch - 3) + (time.time() - t0)) / (epoch - 2)

            train_acc = accuracy(logits[train_mask], labels[train_mask])

            if args.fastmode:
                val_acc = accuracy(logits[val_mask], labels[val_mask])
            else:
                val_acc = evaluate(g, model, features, labels, val_mask)
                if args.early_stop:
                    if stopper.step(val_acc, model):
                        break

            print(
                "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".format(
                    epoch,
                    mean,
                    loss.item(),
                    train_acc,
                    val_acc,
                    n_edges / mean / 1000,
                )
            )

    print()
    if args.early_stop:
        model.load_state_dict(torch.load("es_checkpoint.pt"))
    acc = evaluate(g, model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAT")
    register_data_args(parser)
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="which GPU to use. Set -1 to use CPU.",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="number of training epochs"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="number of hidden attention heads",
    )
    parser.add_argument(
        "--num-out-heads",
        type=int,
        default=1,
        help="number of output attention heads",
    )
    parser.add_argument(
        "--num-layers", type=int, default=1, help="number of hidden layers"
    )
    parser.add_argument(
        "--num-hidden", type=int, default=8, help="number of hidden units"
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        default=False,
        help="use residual connection",
    )
    parser.add_argument(
        "--in-drop", type=float, default=0.7, help="input feature dropout"
    )
    parser.add_argument(
        "--attn-drop", type=float, default=0.7, help="attention dropout"
    )
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="weight decay"
    )
    parser.add_argument(
        "--negative-slope",
        type=float,
        default=0.2,
        help="the negative slope of leaky relu",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        default=False,
        help="indicates whether to use early stop or not",
    )
    parser.add_argument(
        "--fastmode",
        action="store_true",
        default=False,
        help="skip re-evaluate the validation set",
    )
    args = parser.parse_args()
    print(args)

    main(args)
