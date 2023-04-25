#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import os
import random
import time

import dgl

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from models import GAT
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

epsilon = 1 - math.log(2)

device = None

dataset = "ogbn-arxiv"
n_node_feats, n_classes = 0, 0


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def load_data(dataset):
    global n_node_feats, n_classes

    data = DglNodePropPredDataset(name=dataset)
    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    graph, labels = data[0]

    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    return graph, labels, train_idx, val_idx, test_idx, evaluator


def preprocess(graph):
    global n_node_feats

    # make bidirected
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    # add self-loop
    print(f"Total edges before adding self-loop {graph.num_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.num_edges()}")

    graph.create_formats_()

    return graph


def gen_model(args):
    if args.use_labels:
        n_node_feats_ = n_node_feats + n_classes
    else:
        n_node_feats_ = n_node_feats

    model = GAT(
        n_node_feats_,
        n_classes,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        activation=F.relu,
        dropout=args.dropout,
        input_drop=args.input_drop,
        attn_drop=args.attn_drop,
        edge_drop=args.edge_drop,
        use_attn_dst=not args.no_attn_dst,
        use_symmetric_norm=args.use_norm,
    )

    return model


def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def add_labels(feat, labels, idx):
    onehot = torch.zeros([feat.shape[0], n_classes], device=device)
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(
    args,
    model,
    graph,
    labels,
    train_idx,
    val_idx,
    test_idx,
    optimizer,
    evaluator,
):
    model.train()

    feat = graph.ndata["feat"]

    if args.use_labels:
        mask = torch.rand(train_idx.shape) < args.mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask = torch.rand(train_idx.shape) < args.mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    pred = model(graph, feat)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([train_pred_idx, val_idx, test_idx])
        for _ in range(args.n_label_iters):
            pred = pred.detach()
            torch.cuda.empty_cache()
            feat[unlabel_idx, -n_classes:] = F.softmax(
                pred[unlabel_idx], dim=-1
            )
            pred = model(graph, feat)

    loss = custom_loss_function(pred[train_pred_idx], labels[train_pred_idx])
    loss.backward()
    optimizer.step()

    return evaluator(pred[train_idx], labels[train_idx]), loss.item()


@torch.no_grad()
def evaluate(
    args, model, graph, labels, train_idx, val_idx, test_idx, evaluator
):
    model.eval()

    feat = graph.ndata["feat"]

    if args.use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred = model(graph, feat)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([val_idx, test_idx])
        for _ in range(args.n_label_iters):
            feat[unlabel_idx, -n_classes:] = F.softmax(
                pred[unlabel_idx], dim=-1
            )
            pred = model(graph, feat)

    train_loss = custom_loss_function(pred[train_idx], labels[train_idx])
    val_loss = custom_loss_function(pred[val_idx], labels[val_idx])
    test_loss = custom_loss_function(pred[test_idx], labels[test_idx])

    return (
        evaluator(pred[train_idx], labels[train_idx]),
        evaluator(pred[val_idx], labels[val_idx]),
        evaluator(pred[test_idx], labels[test_idx]),
        train_loss,
        val_loss,
        test_loss,
        pred,
    )


def run(
    args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running
):
    evaluator_wrapper = lambda pred, labels: evaluator.eval(
        {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
    )["acc"]

    # define model and optimizer
    model = gen_model(args).to(device)
    optimizer = optim.RMSprop(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )

    # training loop
    total_time = 0
    best_val_acc, final_test_acc, best_val_loss = 0, 0, float("inf")
    final_pred = None

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        adjust_learning_rate(optimizer, args.lr, epoch)

        acc, loss = train(
            args,
            model,
            graph,
            labels,
            train_idx,
            val_idx,
            test_idx,
            optimizer,
            evaluator_wrapper,
        )

        (
            train_acc,
            val_acc,
            test_acc,
            train_loss,
            val_loss,
            test_loss,
            pred,
        ) = evaluate(
            args,
            model,
            graph,
            labels,
            train_idx,
            val_idx,
            test_idx,
            evaluator_wrapper,
        )

        toc = time.time()
        total_time += toc - tic

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            final_test_acc = test_acc
            final_pred = pred

        if epoch == args.n_epochs or epoch % args.log_every == 0:
            print(
                f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}\n"
                f"Loss: {loss:.4f}, Acc: {acc:.4f}\n"
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
            )

        for l, e in zip(
            [
                accs,
                train_accs,
                val_accs,
                test_accs,
                losses,
                train_losses,
                val_losses,
                test_losses,
            ],
            [
                acc,
                train_acc,
                val_acc,
                test_acc,
                loss,
                train_loss,
                val_loss,
                test_loss,
            ],
        ):
            l.append(e)

    print("*" * 50)
    print(f"Best val acc: {best_val_acc}, Final test acc: {final_test_acc}")
    print("*" * 50)

    # plot learning curves
    if args.plot_curves:
        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.set_yticks(np.linspace(0, 1.0, 101))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip(
            [accs, train_accs, val_accs, test_accs],
            ["acc", "train acc", "val acc", "test acc"],
        ):
            plt.plot(range(args.n_epochs), y, label=label, linewidth=1)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gat_acc_{n_running}.png")

        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip(
            [losses, train_losses, val_losses, test_losses],
            ["loss", "train loss", "val loss", "test loss"],
        ):
            plt.plot(range(args.n_epochs), y, label=label, linewidth=1)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gat_loss_{n_running}.png")

    if args.save_pred:
        os.makedirs("./output", exist_ok=True)
        torch.save(F.softmax(final_pred, dim=1), f"./output/{n_running}.pt")

    return best_val_acc, final_test_acc


def count_parameters(args):
    model = gen_model(args)
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def main():
    global device, n_node_feats, n_classes, epsilon

    argparser = argparse.ArgumentParser(
        "GAT implementation on ogbn-arxiv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument(
        "--cpu",
        action="store_true",
        help="CPU mode. This option overrides --gpu.",
    )
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--seed", type=int, default=0, help="seed")
    argparser.add_argument(
        "--n-runs", type=int, default=10, help="running times"
    )
    argparser.add_argument(
        "--n-epochs", type=int, default=2000, help="number of epochs"
    )
    argparser.add_argument(
        "--use-labels",
        action="store_true",
        help="Use labels in the training set as input features.",
    )
    argparser.add_argument(
        "--n-label-iters",
        type=int,
        default=0,
        help="number of label iterations",
    )
    argparser.add_argument(
        "--mask-rate", type=float, default=0.5, help="mask rate"
    )
    argparser.add_argument(
        "--no-attn-dst", action="store_true", help="Don't use attn_dst."
    )
    argparser.add_argument(
        "--use-norm",
        action="store_true",
        help="Use symmetrically normalized adjacency matrix.",
    )
    argparser.add_argument(
        "--lr", type=float, default=0.002, help="learning rate"
    )
    argparser.add_argument(
        "--n-layers", type=int, default=3, help="number of layers"
    )
    argparser.add_argument(
        "--n-heads", type=int, default=3, help="number of heads"
    )
    argparser.add_argument(
        "--n-hidden", type=int, default=250, help="number of hidden units"
    )
    argparser.add_argument(
        "--dropout", type=float, default=0.75, help="dropout rate"
    )
    argparser.add_argument(
        "--input-drop", type=float, default=0.1, help="input drop rate"
    )
    argparser.add_argument(
        "--attn-drop", type=float, default=0.0, help="attention drop rate"
    )
    argparser.add_argument(
        "--edge-drop", type=float, default=0.0, help="edge drop rate"
    )
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument(
        "--log-every", type=int, default=20, help="log every LOG_EVERY epochs"
    )
    argparser.add_argument(
        "--plot-curves", action="store_true", help="plot learning curves"
    )
    argparser.add_argument(
        "--save-pred", action="store_true", help="save final predictions"
    )
    args = argparser.parse_args()

    if not args.use_labels and args.n_label_iters > 0:
        raise ValueError(
            "'--use-labels' must be enabled when n_label_iters > 0"
        )

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    # load data & preprocess
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset)
    graph = preprocess(graph)

    graph, labels, train_idx, val_idx, test_idx = map(
        lambda x: x.to(device), (graph, labels, train_idx, val_idx, test_idx)
    )

    # run
    val_accs, test_accs = [], []

    for i in range(args.n_runs):
        seed(args.seed + i)
        val_acc, test_acc = run(
            args, graph, labels, train_idx, val_idx, test_idx, evaluator, i + 1
        )
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    print(args)
    print(f"Runned {args.n_runs} times")
    print("Val Accs:", val_accs)
    print("Test Accs:", test_accs)
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
    print(f"Number of params: {count_parameters(args)}")


if __name__ == "__main__":
    main()


# Namespace(attn_drop=0.0, cpu=False, dropout=0.75, edge_drop=0.1, gpu=0, input_drop=0.1, log_every=20, lr=0.002, n_epochs=2000, n_heads=3, n_hidden=250, n_label_iters=0, n_layers=3, n_runs=10, no_attn_dst=True, plot_curves=True, use_labels=True, use_norm=True, wd=0)
# Runned 10 times
# Val Accs: [0.7492868888217725, 0.7524413570925199, 0.7505620993993087, 0.7500251686298198, 0.7501929594952851, 0.7513003792073559, 0.7516695191113796, 0.7505285412262156, 0.7504949830531226, 0.7515017282459143]
# Test Accs: [0.7366829208073575, 0.7384112091846182, 0.7368886694236981, 0.7345019854741477, 0.7373001666563792, 0.7362508487130424, 0.7352221056313396, 0.736477172191017, 0.7380614365368393, 0.7362919984363105]
# Average val accuracy: 0.7508003624282694 ± 0.0008760483047616948
# Average test accuracy: 0.736608851305475 ± 0.0011192876013651112
# Number of params: 1441580

# Namespace(attn_drop=0.0, cpu=False, dropout=0.75, edge_drop=0.3, gpu=0, input_drop=0.25, log_every=20, lr=0.002, n_epochs=2000, n_heads=3, n_hidden=250, n_label_iters=1, n_layers=3, n_runs=10, no_attn_dst=True, plot_curves=True, use_labels=True, use_norm=True, wd=0)
# Runned 20 times
# Val Accs: [0.7529782878620088, 0.7521393335346823, 0.7521728917077755, 0.7504949830531226, 0.7518037518037518, 0.7518373099768448, 0.7516359609382866, 0.7511325883418907, 0.7509312393033323, 0.7515017282459143, 0.7511325883418907, 0.7514346118997282, 0.7509312393033323, 0.7521393335346823, 0.7528776133427296, 0.7522735662270545, 0.7504949830531226, 0.7522735662270545, 0.7511661465149837, 0.7501258431490989]
# Test Accs: [0.7390901796185421, 0.7398720243606361, 0.7394605271279551, 0.7384523589078863, 0.7388638561405675, 0.7397280003291978, 0.7414151389831903, 0.7376499393041582, 0.7399748986688065, 0.7400366232537087, 0.7392547785116145, 0.7388844310022015, 0.7374853404110857, 0.7384317840462523, 0.7418677859391396, 0.737937987367035, 0.7381643108450096, 0.7399543238071724, 0.7377322387506944, 0.7385758080776906]
# Average val accuracy: 0.7515738783180644 ± 0.0007617982474634186
# Average test accuracy: 0.7391416167726272 ± 0.0011522198067958794
# Number of params: 1441580
