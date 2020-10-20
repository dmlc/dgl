#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time

import dgl.function as fn
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from torch import nn

from models import GAT

device = None
dataset = "ogbn-proteins"
n_node_feats, n_edge_feats, n_classes = 0, 8, 112


def load_data(dataset):
    data = DglNodePropPredDataset(name=dataset)
    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    graph.ndata["labels"] = labels

    return graph, labels, train_idx, val_idx, test_idx, evaluator


def preprocess(graph, labels):
    global n_node_feats
    # The sum of the weights of adjacent edges is used as the node feature.
    graph.update_all(fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat"))

    n_node_feats = graph.ndata["feat"].shape[-1]

    return graph, labels


def gen_model(args):
    n_node_feats_ = n_node_feats

    model = GAT(
        n_node_feats_,
        n_edge_feats,
        n_classes,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_hidden=args.n_hidden,
        edge_emb=16,
        activation=F.relu,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
    )

    return model


def compute_score(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred, "y_true": labels})["rocauc"]


def train(args, model, graph, _labels, train_idx, criterion, optimizer, _evaluator):
    n_blocks = 10
    batch_size = (len(train_idx) + n_blocks - 1) // n_blocks
    # batch_size = len(train_idx)
    sampler = MultiLayerNeighborSampler([15 for _ in range(args.n_layers)])
    # sampler = MultiLayerFullNeighborSampler(args.n_layers)
    dataloader = NodeDataLoader(
        graph.cpu(), train_idx.cpu(), sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2
    )

    model.train()

    loss_sum, total = 0, 0

    for _input_nodes, output_nodes, subgraphs in dataloader:
        subgraphs = [b.to(device) for b in subgraphs]
        new_train_idx = list(range(len(output_nodes)))

        pred = model(subgraphs)
        loss = criterion(pred[new_train_idx], subgraphs[-1].dstdata["labels"][new_train_idx].float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count = len(new_train_idx)
        loss_sum += loss.item() * count
        total += count

        del loss, pred

    return loss_sum / total


@torch.no_grad()
def evaluate(args, model, graph, labels, train_idx, val_idx, test_idx, criterion, evaluator):
    sampler = MultiLayerNeighborSampler([63 for _ in range(args.n_layers)])
    # sampler = MultiLayerFullNeighborSampler(args.n_layers)
    dataloader = NodeDataLoader(
        graph.cpu(),
        torch.cat([train_idx.cpu(), val_idx.cpu(), test_idx.cpu()]),
        sampler,
        batch_size=32768,
        shuffle=False,
        drop_last=False,
        num_workers=2,
    )

    model.eval()

    preds = torch.zeros(labels.shape).to(device)

    eval_times = 1

    # Due to the limitation of memory capacity, we calculate the average of logits 'eval_times' times.
    for _ in range(eval_times):
        for _input_nodes, output_nodes, subgraphs in dataloader:
            subgraphs = [b.to(device) for b in subgraphs]

            pred = model(subgraphs)
            preds[output_nodes] += pred.detach()

    preds /= eval_times

    train_loss = criterion(preds[train_idx], labels[train_idx].float()).item()
    val_loss = criterion(preds[val_idx], labels[val_idx].float()).item()
    test_loss = criterion(preds[test_idx], labels[test_idx].float()).item()

    return (
        compute_score(preds[train_idx], labels[train_idx], evaluator),
        compute_score(preds[val_idx], labels[val_idx], evaluator),
        compute_score(preds[test_idx], labels[test_idx], evaluator),
        train_loss,
        val_loss,
        test_loss,
    )


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running):
    criterion = nn.BCEWithLogitsLoss()

    model = gen_model(args).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.75, patience=50, verbose=True)

    total_time = 0
    val_score, best_val_score, final_test_score = 0, 0, 0

    train_scores, val_scores, test_scores = [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        loss = train(args, model, graph, labels, train_idx, criterion, optimizer, evaluator)

        toc = time.time()
        total_time += toc - tic

        if epoch % args.eval_every == 0 or epoch % args.log_every == 0:
            train_score, val_score, test_score, train_loss, val_loss, test_loss = evaluate(
                args, model, graph, labels, train_idx, val_idx, test_idx, criterion, evaluator
            )

            if val_score > best_val_score:
                best_val_score = val_score
                final_test_score = test_score

            if epoch % args.log_every == 0:
                print(f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}")
                print(
                    f"Loss: {loss:.4f}\n"
                    f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                    f"Train/Val/Test/Best val/Final test score: {train_score:.4f}/{val_score:.4f}/{test_score:.4f}/{best_val_score:.4f}/{final_test_score:.4f}"
                )

            for l, e in zip(
                [train_scores, val_scores, test_scores, losses, train_losses, val_losses, test_losses],
                [train_score, val_score, test_score, loss, train_loss, val_loss, test_loss],
            ):
                l.append(e)

        lr_scheduler.step(val_score)

    print("*" * 50)
    print(f"Average epoch time: {total_time / args.n_epochs}, Test score: {final_test_score}")

    if args.plot_curves:
        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.set_yticks(np.linspace(0, 1.0, 101))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip([train_scores, val_scores, test_scores], ["train score", "val score", "test score"]):
            plt.plot(range(1, args.n_epochs + 1, args.log_every), y, label=label, linewidth=1)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gat_score_{n_running}.png")

        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip(
            [losses, train_losses, val_losses, test_losses], ["loss", "train loss", "val loss", "test loss"]
        ):
            plt.plot(range(1, args.n_epochs + 1, args.log_every), y, label=label, linewidth=1)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gat_loss_{n_running}.png")

    return best_val_score, final_test_score


def count_parameters(args):
    model = gen_model(args)
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def main():
    global device

    argparser = argparse.ArgumentParser("GAT on OGBN-Proteins", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides '--gpu'.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--n-runs", type=int, default=6)
    argparser.add_argument("--n-epochs", type=int, default=1200)
    argparser.add_argument("--n-heads", type=int, default=15)
    argparser.add_argument("--lr", type=float, default=0.01)
    argparser.add_argument("--n-layers", type=int, default=6)
    argparser.add_argument("--n-hidden", type=int, default=32)
    argparser.add_argument("--dropout", type=float, default=0.3)
    argparser.add_argument("--attn-dropout", type=float, default=0.1)
    argparser.add_argument("--wd", type=float, default=0)
    argparser.add_argument("--eval-every", type=int, default=5)
    argparser.add_argument("--log-every", type=int, default=5)
    argparser.add_argument("--plot-curves", action="store_true")
    args = argparser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % args.gpu)

    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset)
    graph, labels = preprocess(graph, labels)
    graph.create_formats_()

    graph = graph.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)

    val_scores, test_scores = [], []

    for i in range(args.n_runs):
        val_score, test_score = run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, i)
        val_scores.append(val_score)
        test_scores.append(test_score)

    print(args)
    print(f"Runned {args.n_runs} times")
    print("Val scores:", val_scores)
    print("Test scores:", test_scores)
    print(f"Average val score: {np.mean(val_scores)} ± {np.std(val_scores)}")
    print(f"Average test score: {np.mean(test_scores)} ± {np.std(test_scores)}")
    print(f"Number of params: {count_parameters(args)}")


if __name__ == "__main__":
    main()

# Average epoch time: 25.529962027311324, Test score: 0.8653145275750485
# Namespace(attn_dropout=0.1, cpu=False, dropout=0.3, eval_every=5, gpu=0, log_every=5, lr=0.01, n_epochs=1000, n_heads=15, n_hidden=32, n_layers=6, n_runs=6, plot_curves=True, wd=0)
# Runned 6 times
# Val scores: [0.9158038128041037, 0.9152629032809712, 0.9156345294978633, 0.9150481914782242, 0.9157425464460424, 0.9161184010407541]
# Test scores: [0.8677984231764927, 0.8641230174511977, 0.8659541569395243, 0.8657108194912686, 0.8626434354220878, 0.8653145275750485]
# Average val score: 0.9156017307579932 ± 0.0003535298061971046
# Average test score: 0.8652573966759366 ± 0.0015953451242433543
# Number of params: 2436304

# Average epoch time: 25.627150499622026, Test score: 0.8645282897120268
# Namespace(attn_dropout=0.1, cpu=False, dropout=0.3, eval_every=5, gpu=0, log_every=5, lr=0.01, n_epochs=1200, n_heads=15, n_hidden=32, n_layers=6, n_runs=6, plot_curves=True, wd=0)
# Runned 6 times
# Val scores: [0.9160934481239426, 0.9154423466083738, 0.9164203656218792, 0.916688827518927, 0.9170246800652017, 0.9160127508532699]
# Test scores: [0.8675282938770733, 0.866573316395541, 0.8669931192520849, 0.8673913226417784, 0.8657328707151425, 0.8645282897120268]
# Average val score: 0.9162804031319323 ± 0.0005081464332659773
# Average test score: 0.8664578687656078 ± 0.0010460931120063052
# Number of params: 2436304
