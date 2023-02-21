""" The main file to train an ARMA model using a full graph """

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from model import ARMA4NC
from tqdm import trange


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

    # check cuda
    device = (
        f"cuda:{args.gpu}"
        if args.gpu >= 0 and torch.cuda.is_available()
        else "cpu"
    )

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
    model = ARMA4NC(
        in_dim=n_features,
        hid_dim=args.hid_dim,
        out_dim=n_classes,
        num_stacks=args.num_stacks,
        num_layers=args.num_layers,
        activation=nn.ReLU(),
        dropout=args.dropout,
    ).to(device)

    best_model = copy.deepcopy(model)

    # Step 3: Create training components ===================================================== #
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lamb)

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

    best_model.eval()
    logits = best_model(graph, feats)
    test_acc = torch.sum(
        logits[test_idx].argmax(dim=1) == labels[test_idx]
    ).item() / len(test_idx)

    print("Test Acc {:.4f}".format(test_acc))
    return test_acc


if __name__ == "__main__":
    """
    ARMA Model Hyperparameters
    """
    parser = argparse.ArgumentParser(description="ARMA GCN")

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
        default=100,
        help="Patient epochs to wait before early stopping.",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--lamb", type=float, default=5e-4, help="L2 reg.")
    # model params
    parser.add_argument(
        "--hid-dim", type=int, default=16, help="Hidden layer dimensionalities."
    )
    parser.add_argument(
        "--num-stacks", type=int, default=2, help="Number of K."
    )
    parser.add_argument(
        "--num-layers", type=int, default=1, help="Number of T."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.75,
        help="Dropout applied at all layers.",
    )

    args = parser.parse_args()
    print(args)

    acc_lists = []

    for _ in range(100):
        acc_lists.append(main(args))

    mean = np.around(np.mean(acc_lists, axis=0), decimals=3)
    std = np.around(np.std(acc_lists, axis=0), decimals=3)
    print("Total acc: ", acc_lists)
    print("mean", mean)
    print("std", std)
