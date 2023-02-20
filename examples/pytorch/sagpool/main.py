import argparse
import json
import logging
import os
from time import time

import dgl

import torch
import torch.nn
import torch.nn.functional as F
from dgl.data import LegacyTUDataset
from dgl.dataloading import GraphDataLoader
from network import get_sag_network
from torch.utils.data import random_split
from utils import get_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Self-Attention Graph Pooling")
    parser.add_argument(
        "--dataset",
        type=str,
        default="DD",
        choices=["DD", "PROTEINS", "NCI1", "NCI109", "Mutagenicity"],
        help="DD/PROTEINS/NCI1/NCI109/Mutagenicity",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="batch size"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="weight decay"
    )
    parser.add_argument(
        "--pool_ratio", type=float, default=0.5, help="pooling ratio"
    )
    parser.add_argument("--hid_dim", type=int, default=128, help="hidden size")
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout ratio"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100000,
        help="max number of training epochs",
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="patience for early stopping"
    )
    parser.add_argument(
        "--device", type=int, default=-1, help="device id, -1 for cpu"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="hierarchical",
        choices=["hierarchical", "global"],
        help="model architecture",
    )
    parser.add_argument(
        "--dataset_path", type=str, default="./dataset", help="path to dataset"
    )
    parser.add_argument(
        "--conv_layers", type=int, default=3, help="number of conv layers"
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="print trainlog every k epochs, -1 for silent training",
    )
    parser.add_argument(
        "--num_trials", type=int, default=1, help="number of trials"
    )
    parser.add_argument("--output_path", type=str, default="./output")

    args = parser.parse_args()

    # device
    args.device = "cpu" if args.device == -1 else "cuda:{}".format(args.device)
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, use CPU for training.")
        args.device = "cpu"

    # print every
    if args.print_every == -1:
        args.print_every = args.epochs + 1

    # paths
    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    name = "Data={}_Hidden={}_Arch={}_Pool={}_WeightDecay={}_Lr={}.log".format(
        args.dataset,
        args.hid_dim,
        args.architecture,
        args.pool_ratio,
        args.weight_decay,
        args.lr,
    )
    args.output_path = os.path.join(args.output_path, name)

    return args


def train(model: torch.nn.Module, optimizer, trainloader, device):
    model.train()
    total_loss = 0.0
    num_batches = len(trainloader)
    for batch in trainloader:
        optimizer.zero_grad()
        batch_graphs, batch_labels = batch
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs)
        loss = F.nll_loss(out, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches


@torch.no_grad()
def test(model: torch.nn.Module, loader, device):
    model.eval()
    correct = 0.0
    loss = 0.0
    num_graphs = 0
    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs)
        pred = out.argmax(dim=1)
        loss += F.nll_loss(out, batch_labels, reduction="sum").item()
        correct += pred.eq(batch_labels).sum().item()
    return correct / num_graphs, loss / num_graphs


def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    dataset = LegacyTUDataset(args.dataset, raw_dir=args.dataset_path)

    # add self loop. We add self loop for each graph here since the function "add_self_loop" does not
    # support batch graph.
    for i in range(len(dataset)):
        dataset.graph_lists[i] = dgl.add_self_loop(dataset.graph_lists[i])

    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - num_val - num_training
    train_set, val_set, test_set = random_split(
        dataset, [num_training, num_val, num_test]
    )

    train_loader = GraphDataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=6
    )
    val_loader = GraphDataLoader(
        val_set, batch_size=args.batch_size, num_workers=2
    )
    test_loader = GraphDataLoader(
        test_set, batch_size=args.batch_size, num_workers=2
    )

    device = torch.device(args.device)

    # Step 2: Create model =================================================================== #
    num_feature, num_classes, _ = dataset.statistics()
    model_op = get_sag_network(args.architecture)
    model = model_op(
        in_dim=num_feature,
        hid_dim=args.hid_dim,
        out_dim=num_classes,
        num_convs=args.conv_layers,
        pool_ratio=args.pool_ratio,
        dropout=args.dropout,
    ).to(device)
    args.num_feature = int(num_feature)
    args.num_classes = int(num_classes)

    # Step 3: Create training components ===================================================== #
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Step 4: training epoches =============================================================== #
    bad_cound = 0
    best_val_loss = float("inf")
    final_test_acc = 0.0
    best_epoch = 0
    train_times = []
    for e in range(args.epochs):
        s_time = time()
        train_loss = train(model, optimizer, train_loader, device)
        train_times.append(time() - s_time)
        val_acc, val_loss = test(model, val_loader, device)
        test_acc, _ = test(model, test_loader, device)
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            final_test_acc = test_acc
            bad_cound = 0
            best_epoch = e + 1
        else:
            bad_cound += 1
        if bad_cound >= args.patience:
            break

        if (e + 1) % args.print_every == 0:
            log_format = (
                "Epoch {}: loss={:.4f}, val_acc={:.4f}, final_test_acc={:.4f}"
            )
            print(log_format.format(e + 1, train_loss, val_acc, final_test_acc))
    print(
        "Best Epoch {}, final test acc {:.4f}".format(
            best_epoch, final_test_acc
        )
    )
    return final_test_acc, sum(train_times) / len(train_times)


if __name__ == "__main__":
    args = parse_args()
    res = []
    train_times = []
    for i in range(args.num_trials):
        print("Trial {}/{}".format(i + 1, args.num_trials))
        acc, train_time = main(args)
        res.append(acc)
        train_times.append(train_time)

    mean, err_bd = get_stats(res)
    print("mean acc: {:.4f}, error bound: {:.4f}".format(mean, err_bd))

    out_dict = {
        "hyper-parameters": vars(args),
        "result": "{:.4f}(+-{:.4f})".format(mean, err_bd),
        "train_time": "{:.4f}".format(sum(train_times) / len(train_times)),
    }

    with open(args.output_path, "w") as f:
        json.dump(out_dict, f, sort_keys=True, indent=4)
