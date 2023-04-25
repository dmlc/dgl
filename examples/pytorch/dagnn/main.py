import argparse

import dgl.function as fn

import numpy as np
import torch
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from torch import nn
from torch.nn import functional as F, Parameter
from tqdm import trange
from utils import evaluate, generate_random_seeds, set_random_state


class DAGNNConv(nn.Module):
    def __init__(self, in_dim, k):
        super(DAGNNConv, self).__init__()

        self.s = Parameter(torch.FloatTensor(in_dim, 1))
        self.k = k

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("sigmoid")
        nn.init.xavier_uniform_(self.s, gain=gain)

    def forward(self, graph, feats):
        with graph.local_scope():
            results = [feats]

            degs = graph.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm = norm.to(feats.device).unsqueeze(1)

            for _ in range(self.k):
                feats = feats * norm
                graph.ndata["h"] = feats
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                feats = graph.ndata["h"]
                feats = feats * norm
                results.append(feats)

            H = torch.stack(results, dim=1)
            S = F.sigmoid(torch.matmul(H, self.s))
            S = S.permute(0, 2, 1)
            H = torch.matmul(S, H).squeeze()

            return H


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation=None, dropout=0):
        super(MLPLayer, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = 1.0
        if self.activation is F.relu:
            gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, feats):
        feats = self.dropout(feats)
        feats = self.linear(feats)
        if self.activation:
            feats = self.activation(feats)

        return feats


class DAGNN(nn.Module):
    def __init__(
        self,
        k,
        in_dim,
        hid_dim,
        out_dim,
        bias=True,
        activation=F.relu,
        dropout=0,
    ):
        super(DAGNN, self).__init__()
        self.mlp = nn.ModuleList()
        self.mlp.append(
            MLPLayer(
                in_dim=in_dim,
                out_dim=hid_dim,
                bias=bias,
                activation=activation,
                dropout=dropout,
            )
        )
        self.mlp.append(
            MLPLayer(
                in_dim=hid_dim,
                out_dim=out_dim,
                bias=bias,
                activation=None,
                dropout=dropout,
            )
        )
        self.dagnn = DAGNNConv(in_dim=out_dim, k=k)

    def forward(self, graph, feats):
        for layer in self.mlp:
            feats = layer(feats)
        feats = self.dagnn(graph, feats)
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
    graph = graph.add_self_loop()

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
    model = DAGNN(
        k=args.k,
        in_dim=n_features,
        hid_dim=args.hid_dim,
        out_dim=n_classes,
        dropout=args.dropout,
    )
    model = model.to(device)

    # Step 3: Create training components ===================================================== #
    loss_fn = F.cross_entropy
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.lamb
    )

    # Step 4: training epochs =============================================================== #
    loss = float("inf")
    best_acc = 0
    no_improvement = 0
    epochs = trange(args.epochs, desc="Accuracy & Loss")

    for _ in epochs:
        model.train()

        logits = model(graph, feats)

        # compute loss
        train_loss = loss_fn(logits[train_idx], labels[train_idx])

        # backward
        opt.zero_grad()
        train_loss.backward()
        opt.step()

        (
            train_loss,
            train_acc,
            valid_loss,
            valid_acc,
            test_loss,
            test_acc,
        ) = evaluate(
            model, graph, feats, labels, (train_idx, val_idx, test_idx)
        )

        # Print out performance
        epochs.set_description(
            "Train Acc {:.4f} | Train Loss {:.4f} | Val Acc {:.4f} | Val loss {:.4f}".format(
                train_acc, train_loss.item(), valid_acc, valid_loss.item()
            )
        )

        if valid_loss > loss:
            no_improvement += 1
            if no_improvement == args.early_stopping:
                print("Early stop.")
                break
        else:
            no_improvement = 0
            loss = valid_loss
            best_acc = test_acc

    print("Test Acc {:.4f}".format(best_acc))
    return best_acc


if __name__ == "__main__":
    """
    DAGNN Model Hyperparameters
    """
    parser = argparse.ArgumentParser(description="DAGNN")
    # data source params
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        choices=["Cora", "Citeseer", "Pubmed"],
        help="Name of dataset.",
    )
    # cuda params
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU index. Default: -1, using CPU."
    )
    # training params
    parser.add_argument("--runs", type=int, default=1, help="Training runs.")
    parser.add_argument(
        "--epochs", type=int, default=1500, help="Training epochs."
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=100,
        help="Patient epochs to wait before early stopping.",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--lamb", type=float, default=0.005, help="L2 reg.")
    # model params
    parser.add_argument(
        "--k", type=int, default=12, help="Number of propagation layers."
    )
    parser.add_argument(
        "--hid-dim", type=int, default=64, help="Hidden layer dimensionalities."
    )
    parser.add_argument("--dropout", type=float, default=0.8, help="dropout")
    args = parser.parse_args()
    print(args)

    acc_lists = []
    random_seeds = generate_random_seeds(seed=1222, nums=args.runs)

    for run in range(args.runs):
        set_random_state(random_seeds[run])
        acc_lists.append(main(args))

    acc_lists = np.array(acc_lists)

    mean = np.around(np.mean(acc_lists, axis=0), decimals=4)
    std = np.around(np.std(acc_lists, axis=0), decimals=4)

    print("Total acc: ", acc_lists)
    print("mean", mean)
    print("std", std)
