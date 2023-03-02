""" The main file to train a MixHop model using a full graph """

import argparse
import copy
import random

import dgl
import dgl.function as fn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from tqdm import trange


class MixHopConv(nn.Module):
    r"""

    Description
    -----------
    MixHop Graph Convolutional layer from paper `MixHop: Higher-Order Graph Convolutional Architecturesvia Sparsified Neighborhood Mixing
     <https://arxiv.org/pdf/1905.00067.pdf>`__.

    .. math::
        H^{(i+1)} =\underset{j \in P}{\Bigg\Vert} \sigma\left(\widehat{A}^j H^{(i)} W_j^{(i)}\right),

    where :math:`\widehat{A}` denotes the symmetrically normalized adjacencymatrix with self-connections,
    :math:`D_{ii} = \sum_{j=0} \widehat{A}_{ij}` its diagonal degree matrix,
    :math:`W_j^{(i)}` denotes the trainable weight matrix of different MixHop layers.

    Parameters
    ----------
    in_dim : int
        Input feature size. i.e, the number of dimensions of :math:`H^{(i)}`.
    out_dim : int
        Output feature size for each power.
    p: list
        List of powers of adjacency matrix. Defaults: ``[0, 1, 2]``.
    dropout: float, optional
        Dropout rate on node features. Defaults: ``0``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    batchnorm: bool, optional
        If True, use batch normalization. Defaults: ``False``.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        p=[0, 1, 2],
        dropout=0,
        activation=None,
        batchnorm=False,
    ):
        super(MixHopConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p = p
        self.activation = activation
        self.batchnorm = batchnorm

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim * len(p))

        # define weight dict for each power j
        self.weights = nn.ModuleDict(
            {str(j): nn.Linear(in_dim, out_dim, bias=False) for j in p}
        )

    def forward(self, graph, feats):
        with graph.local_scope():
            # assume that the graphs are undirected and graph.in_degrees() is the same as graph.out_degrees()
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)
            max_j = max(self.p) + 1
            outputs = []
            for j in range(max_j):
                if j in self.p:
                    output = self.weights[str(j)](feats)
                    outputs.append(output)

                feats = feats * norm
                graph.ndata["h"] = feats
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                feats = graph.ndata.pop("h")
                feats = feats * norm

            final = torch.cat(outputs, dim=1)

            if self.batchnorm:
                final = self.bn(final)

            if self.activation is not None:
                final = self.activation(final)

            final = self.dropout(final)

            return final


class MixHop(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        num_layers=2,
        p=[0, 1, 2],
        input_dropout=0.0,
        layer_dropout=0.0,
        activation=None,
        batchnorm=False,
    ):
        super(MixHop, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.p = p
        self.input_dropout = input_dropout
        self.layer_dropout = layer_dropout
        self.activation = activation
        self.batchnorm = batchnorm

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(self.input_dropout)

        # Input layer
        self.layers.append(
            MixHopConv(
                self.in_dim,
                self.hid_dim,
                p=self.p,
                dropout=self.input_dropout,
                activation=self.activation,
                batchnorm=self.batchnorm,
            )
        )

        # Hidden layers with n - 1 MixHopConv layers
        for i in range(self.num_layers - 2):
            self.layers.append(
                MixHopConv(
                    self.hid_dim * len(args.p),
                    self.hid_dim,
                    p=self.p,
                    dropout=self.layer_dropout,
                    activation=self.activation,
                    batchnorm=self.batchnorm,
                )
            )

        self.fc_layers = nn.Linear(
            self.hid_dim * len(args.p), self.out_dim, bias=False
        )

    def forward(self, graph, feats):
        feats = self.dropout(feats)
        for layer in self.layers:
            feats = layer(graph, feats)

        feats = self.fc_layers(feats)

        return feats


def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # Load from DGL dataset
    if args.dataset == "Cora":
        dataset = CoraGraphDataset()
    elif args.dataset == "Citeseer":
        dataset = CiteseerGraphDataset()
    elif args.dataset == "Pubmed":
        dataset = PubmedGraphDataset()
    else:
        raise ValueError("Dataset {} is invalid.".format(args.dataset))

    graph = dataset[0]
    graph = dgl.add_self_loop(graph)

    # check cuda
    if args.gpu >= 0 and torch.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    # retrieve the number of classes
    n_classes = dataset.num_classes

    # retrieve labels of ground truth
    labels = graph.ndata.pop("label").to(device).long()

    # Extract node features
    feats = graph.ndata.pop("feat").to(device)
    n_features = feats.shape[-1]

    # retrieve masks for train/validation/test
    train_mask = graph.ndata.pop("train_mask")
    val_mask = graph.ndata.pop("val_mask")
    test_mask = graph.ndata.pop("test_mask")

    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze().to(device)
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze().to(device)
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze().to(device)

    graph = graph.to(device)

    # Step 2: Create model =================================================================== #
    model = MixHop(
        in_dim=n_features,
        hid_dim=args.hid_dim,
        out_dim=n_classes,
        num_layers=args.num_layers,
        p=args.p,
        input_dropout=args.input_dropout,
        layer_dropout=args.layer_dropout,
        activation=torch.tanh,
        batchnorm=True,
    )

    model = model.to(device)
    best_model = copy.deepcopy(model)

    # Step 3: Create training components ===================================================== #
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lamb)
    scheduler = optim.lr_scheduler.StepLR(opt, args.step_size, gamma=args.gamma)

    # Step 4: training epoches =============================================================== #
    acc = 0
    no_improvement = 0
    epochs = trange(args.epochs, desc="Accuracy & Loss")

    for _ in epochs:
        # Training using a full graph
        model.train()

        logits = model(graph, feats)

        # compute loss
        train_loss = loss_fn(logits[train_idx], labels[train_idx])
        train_acc = torch.sum(
            logits[train_idx].argmax(dim=1) == labels[train_idx]
        ).item() / len(train_idx)

        # backward
        opt.zero_grad()
        train_loss.backward()
        opt.step()

        # Validation using a full graph
        model.eval()

        with torch.no_grad():
            valid_loss = loss_fn(logits[val_idx], labels[val_idx])
            valid_acc = torch.sum(
                logits[val_idx].argmax(dim=1) == labels[val_idx]
            ).item() / len(val_idx)

        # Print out performance
        epochs.set_description(
            "Train Acc {:.4f} | Train Loss {:.4f} | Val Acc {:.4f} | Val loss {:.4f}".format(
                train_acc, train_loss.item(), valid_acc, valid_loss.item()
            )
        )

        if valid_acc < acc:
            no_improvement += 1
            if no_improvement == args.early_stopping:
                print("Early stop.")
                break
        else:
            no_improvement = 0
            acc = valid_acc
            best_model = copy.deepcopy(model)

        scheduler.step()

    best_model.eval()
    logits = best_model(graph, feats)
    test_acc = torch.sum(
        logits[test_idx].argmax(dim=1) == labels[test_idx]
    ).item() / len(test_idx)

    print("Test Acc {:.4f}".format(test_acc))
    return test_acc


if __name__ == "__main__":
    """
    MixHop Model Hyperparameters
    """
    parser = argparse.ArgumentParser(description="MixHop GCN")

    # data source params
    parser.add_argument(
        "--dataset", type=str, default="Cora", help="Name of dataset."
    )
    # cuda params
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU index. Default: -1, using CPU."
    )
    # training params
    parser.add_argument(
        "--epochs", type=int, default=2000, help="Training epochs."
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=200,
        help="Patient epochs to wait before early stopping.",
    )
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate.")
    parser.add_argument("--lamb", type=float, default=5e-4, help="L2 reg.")
    parser.add_argument(
        "--step-size",
        type=int,
        default=40,
        help="Period of learning rate decay.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.01,
        help="Multiplicative factor of learning rate decay.",
    )
    # model params
    parser.add_argument(
        "--hid-dim", type=int, default=60, help="Hidden layer dimensionalities."
    )
    parser.add_argument(
        "--num-layers", type=int, default=4, help="Number of GNN layers."
    )
    parser.add_argument(
        "--input-dropout",
        type=float,
        default=0.7,
        help="Dropout applied at input layer.",
    )
    parser.add_argument(
        "--layer-dropout",
        type=float,
        default=0.9,
        help="Dropout applied at hidden layers.",
    )
    parser.add_argument(
        "--p", nargs="+", type=int, help="List of powers of adjacency matrix."
    )

    parser.set_defaults(p=[0, 1, 2])

    args = parser.parse_args()
    print(args)

    acc_lists = []

    for _ in range(100):
        acc_lists.append(main(args))

    acc_lists.sort()
    acc_lists_top = np.array(acc_lists[50:])

    mean = np.around(np.mean(acc_lists_top, axis=0), decimals=3)
    std = np.around(np.std(acc_lists_top, axis=0), decimals=3)
    print("Total acc: ", acc_lists)
    print("Top 50 acc:", acc_lists_top)
    print("mean", mean)
    print("std", std)
