#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import random
import time
from collections import OrderedDict

import dgl.function as fn

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    MultiLayerNeighborSampler,
)
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from models import MLP
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from torch import nn
from tqdm import tqdm

epsilon = 1 - math.log(2)

device = None
dataset = "ogbn-products"
n_node_feats, n_edge_feats, n_classes = 0, 0, 0


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(dataset):
    data = DglNodePropPredDataset(name=dataset)
    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    graph, labels = data[0]
    graph.ndata["labels"] = labels

    return graph, labels, train_idx, val_idx, test_idx, evaluator


def preprocess(graph, labels):
    global n_node_feats, n_classes
    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    # graph = graph.remove_self_loop().add_self_loop()
    n_node_feats = graph.ndata["feat"].shape[-1]

    return graph, labels


def gen_model(args):
    model = MLP(
        n_node_feats,
        n_classes,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        activation=F.relu,
        dropout=args.dropout,
        input_drop=args.input_drop,
        residual=False,
    )

    return model


def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def train(
    args, model, dataloader, labels, train_idx, criterion, optimizer, evaluator
):
    model.train()

    loss_sum, total = 0, 0

    preds = torch.zeros(labels.shape[0], n_classes)

    with dataloader.enable_cpu_affinity():
        for _input_nodes, output_nodes, subgraphs in dataloader:
            subgraphs = [b.to(device) for b in subgraphs]
            new_train_idx = list(range(len(output_nodes)))

            pred = model(subgraphs[0].srcdata["feat"])
            preds[output_nodes] = pred.cpu().detach()

            loss = criterion(
                pred[new_train_idx], labels[output_nodes][new_train_idx]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = len(new_train_idx)
            loss_sum += loss.item() * count
            total += count

    preds = preds.to(train_idx.device)
    return (
        loss_sum / total,
        evaluator(preds[train_idx], labels[train_idx]),
    )


@torch.no_grad()
def evaluate(
    args,
    model,
    dataloader,
    labels,
    train_idx,
    val_idx,
    test_idx,
    criterion,
    evaluator,
):
    model.eval()

    preds = torch.zeros(labels.shape[0], n_classes, device=device)

    eval_times = 1  # Due to the limitation of memory capacity, we calculate the average of logits 'eval_times' times.

    for _ in range(eval_times):
        with dataloader.enable_cpu_affinity():
            for _input_nodes, output_nodes, subgraphs in dataloader:
                subgraphs = [b.to(device) for b in subgraphs]

                pred = model(subgraphs[0].srcdata["feat"])
                preds[output_nodes] = pred

    preds /= eval_times

    train_loss = criterion(preds[train_idx], labels[train_idx]).item()
    val_loss = criterion(preds[val_idx], labels[val_idx]).item()
    test_loss = criterion(preds[test_idx], labels[test_idx]).item()

    return (
        evaluator(preds[train_idx], labels[train_idx]),
        evaluator(preds[val_idx], labels[val_idx]),
        evaluator(preds[test_idx], labels[test_idx]),
        train_loss,
        val_loss,
        test_loss,
    )


def run(
    args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running
):
    evaluator_wrapper = lambda pred, labels: evaluator.eval(
        {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
    )["acc"]
    criterion = custom_loss_function

    train_batch_size = 4096
    train_sampler = MultiLayerNeighborSampler(
        [0 for _ in range(args.n_layers)]
    )  # no not sample neighbors
    train_dataloader = DataLoader(
        graph.cpu(),
        train_idx.cpu(),
        train_sampler,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
    )

    eval_batch_size = 4096
    eval_sampler = MultiLayerNeighborSampler(
        [0 for _ in range(args.n_layers)]
    )  # no not sample neighbors
    if args.eval_last:
        eval_idx = torch.cat([train_idx.cpu(), val_idx.cpu()])
    else:
        eval_idx = torch.cat([train_idx.cpu(), val_idx.cpu(), test_idx.cpu()])
    eval_dataloader = DataLoader(
        graph.cpu(),
        eval_idx,
        eval_sampler,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=4,
    )

    model = gen_model(args).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.7,
        patience=20,
        min_lr=1e-4,
    )

    best_model_state_dict = None

    total_time = 0
    val_score, best_val_score, final_test_score = 0, 0, 0

    scores, train_scores, val_scores, test_scores = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()
        loss, score = train(
            args,
            model,
            train_dataloader,
            labels,
            train_idx,
            criterion,
            optimizer,
            evaluator_wrapper,
        )

        toc = time.time()
        total_time += toc - tic

        if epoch % args.eval_every == 0 or epoch % args.log_every == 0:
            (
                train_score,
                val_score,
                test_score,
                train_loss,
                val_loss,
                test_loss,
            ) = evaluate(
                args,
                model,
                eval_dataloader,
                labels,
                train_idx,
                val_idx,
                test_idx,
                criterion,
                evaluator_wrapper,
            )

            if val_score > best_val_score:
                best_val_score = val_score
                final_test_score = test_score
                if args.eval_last:
                    best_model_state_dict = {
                        k: v.to("cpu") for k, v in model.state_dict().items()
                    }
                    best_model_state_dict = OrderedDict(best_model_state_dict)

            if epoch % args.log_every == 0:
                print(
                    f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch}"
                )
                print(
                    f"Loss: {loss:.4f}, Score: {score:.4f}\n"
                    f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                    f"Train/Val/Test/Best val/Final test score: {train_score:.4f}/{val_score:.4f}/{test_score:.4f}/{best_val_score:.4f}/{final_test_score:.4f}"
                )

            for l, e in zip(
                [
                    scores,
                    train_scores,
                    val_scores,
                    test_scores,
                    losses,
                    train_losses,
                    val_losses,
                    test_losses,
                ],
                [
                    score,
                    train_score,
                    val_score,
                    test_score,
                    loss,
                    train_loss,
                    val_loss,
                    test_loss,
                ],
            ):
                l.append(e)

        lr_scheduler.step(val_score)

    if args.eval_last:
        model.load_state_dict(best_model_state_dict)
        eval_dataloader = DataLoader(
            graph.cpu(),
            test_idx.cpu(),
            eval_sampler,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=4,
        )
        final_test_score = evaluate(
            args,
            model,
            eval_dataloader,
            labels,
            train_idx,
            val_idx,
            test_idx,
            criterion,
            evaluator_wrapper,
        )[2]

    print("*" * 50)
    print(
        f"Average epoch time: {total_time / args.n_epochs}, Test score: {final_test_score}"
    )

    if args.plot_curves:
        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.set_yticks(np.linspace(0, 1.0, 101))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip(
            [train_scores, val_scores, test_scores],
            ["train score", "val score", "test score"],
        ):
            plt.plot(
                range(1, args.n_epochs + 1, args.log_every),
                y,
                label=label,
                linewidth=1,
            )
        ax.xaxis.set_major_locator(MultipleLocator(20))
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
            [losses, train_losses, val_losses, test_losses],
            ["loss", "train loss", "val loss", "test loss"],
        ):
            plt.plot(
                range(1, args.n_epochs + 1, args.log_every),
                y,
                label=label,
                linewidth=1,
            )
        ax.xaxis.set_major_locator(MultipleLocator(20))
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
    return sum(
        [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
    )


def main():
    global device

    argparser = argparse.ArgumentParser(
        "GAT on OGBN-Proteins",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument(
        "--cpu",
        action="store_true",
        help="CPU mode. This option overrides '--gpu'.",
    )
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--seed", type=int, help="seed", default=0)
    argparser.add_argument("--n-runs", type=int, default=10)
    argparser.add_argument("--n-epochs", type=int, default=500)
    argparser.add_argument("--lr", type=float, default=0.01)
    argparser.add_argument("--n-layers", type=int, default=4)
    argparser.add_argument("--n-hidden", type=int, default=480)
    argparser.add_argument("--dropout", type=float, default=0.2)
    argparser.add_argument("--input-drop", type=float, default=0)
    argparser.add_argument("--wd", type=float, default=0)
    argparser.add_argument(
        "--estimation-mode",
        action="store_true",
        help="Estimate the score of test set for speed.",
    )
    argparser.add_argument(
        "--eval-last",
        action="store_true",
        help="Evaluate the score of test set at last.",
    )
    argparser.add_argument("--eval-every", type=int, default=1)
    argparser.add_argument("--log-every", type=int, default=1)
    argparser.add_argument("--plot-curves", action="store_true")
    args = argparser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % args.gpu)

    if args.estimation_mode:
        print(
            "WARNING: Estimation mode is enabled. The test score is not accurate."
        )

    seed(args.seed)

    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset)
    graph, labels = preprocess(graph, labels)
    graph.create_formats_()

    # graph = graph.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    if args.estimation_mode:
        test_idx = test_idx[torch.arange(start=0, end=len(test_idx), step=50)]

    val_scores, test_scores = [], []

    for i in range(1, args.n_runs + 1):
        val_score, test_score = run(
            args, graph, labels, train_idx, val_idx, test_idx, evaluator, i
        )
        val_scores.append(val_score)
        test_scores.append(test_score)

    print(args)
    print(f"Runned {args.n_runs} times")
    print("Val scores:", val_scores)
    print("Test scores:", test_scores)
    print(f"Average val score: {np.mean(val_scores)} ± {np.std(val_scores)}")
    print(f"Average test score: {np.mean(test_scores)} ± {np.std(test_scores)}")
    print(f"Number of params: {count_parameters(args)}")

    if args.estimation_mode:
        print(
            "WARNING: Estimation mode is enabled. The test score is not accurate."
        )


if __name__ == "__main__":
    main()

# Namespace(cpu=False, dropout=0.2, estimation_mode=False, eval_every=1, eval_last=True, gpu=2, input_drop=0, log_every=1, lr=0.01, n_epochs=500, n_hidden=480, n_layers=4, n_runs=10, plot_curves=True, seed=0, wd=0)
# Runned 10 times
# Val scores: [0.7846298603870508, 0.7811713246700405, 0.7828751621188618, 0.7839941001449533, 0.7843501258805279, 0.7841466826030568, 0.7846298603870508, 0.7865880019327112, 0.7832057574447524, 0.7851384685807289]
# Test scores: [0.6318660190656417, 0.6304137516261193, 0.6329961126767946, 0.6312885462007662, 0.6340624944929965, 0.6301507710256831, 0.6314534738969161, 0.6334637843631373, 0.6312465235275007, 0.6329857199726536]
# Average val score: 0.7840729344149735 ± 0.0013702460721628086
# Average test score: 0.6319927196848208 ± 0.001252448369121226
# Number of params: 535727
