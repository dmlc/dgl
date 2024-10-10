#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import random
import time
from collections import OrderedDict

import dgl
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
from models import GAT
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
    dgl.random.seed(seed)


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


def preprocess(graph, labels, train_idx):
    global n_node_feats, n_classes
    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    # graph = graph.remove_self_loop().add_self_loop()
    n_node_feats = graph.ndata["feat"].shape[-1]

    graph.ndata["train_labels_onehot"] = torch.zeros(
        graph.num_nodes(), n_classes
    )
    graph.ndata["train_labels_onehot"][train_idx, labels[train_idx, 0]] = 1

    graph.ndata["is_train"] = torch.zeros(graph.num_nodes(), dtype=torch.bool)
    graph.ndata["is_train"][train_idx] = 1

    graph.create_formats_()

    return graph, labels


def gen_model(args):
    if args.use_labels:
        n_node_feats_ = n_node_feats + n_classes
    else:
        n_node_feats_ = n_node_feats

    model = GAT(
        n_node_feats_,
        n_edge_feats,
        n_classes,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_hidden=args.n_hidden,
        edge_emb=0,
        activation=F.relu,
        dropout=args.dropout,
        input_drop=args.input_drop,
        attn_drop=args.attn_dropout,
        edge_drop=args.edge_drop,
        use_attn_dst=not args.no_attn_dst,
        allow_zero_in_degree=True,
        residual=False,
    )

    return model


def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def add_soft_labels(graph, soft_labels):
    feat = graph.srcdata["feat"]
    graph.srcdata["feat"] = torch.cat([feat, soft_labels], dim=-1)


def update_hard_labels(graph, idx=None):
    if idx is None:
        idx = torch.arange(graph.srcdata["is_train"].shape[0])[
            graph.srcdata["is_train"]
        ]

    graph.srcdata["feat"][idx, -n_classes:] = graph.srcdata[
        "train_labels_onehot"
    ][idx]


def train(
    args, model, dataloader, labels, train_idx, criterion, optimizer, evaluator
):
    model.train()

    loss_sum, total = 0, 0

    preds = torch.zeros(labels.shape[0], n_classes)

    for it in range(args.n_label_iters + 1):
        preds_old = preds.clone()
        for input_nodes, output_nodes, subgraphs in dataloader:
            subgraphs = [b.to(device) for b in subgraphs]
            new_train_idx = torch.arange(len(output_nodes))

            if args.use_labels:
                mask = torch.rand(new_train_idx.shape) < args.mask_rate

                train_labels_idx = torch.cat(
                    [
                        new_train_idx[~mask],
                        torch.arange(len(output_nodes), len(input_nodes)),
                    ]
                )
                train_pred_idx = new_train_idx[mask]

                add_soft_labels(
                    subgraphs[0],
                    F.softmax(preds_old[input_nodes].to(device), dim=-1),
                )
                update_hard_labels(subgraphs[0], train_labels_idx)
            else:
                train_pred_idx = new_train_idx

            pred = model(subgraphs)

            preds[output_nodes] = pred.cpu().detach()

            # NOTE: This is not a complete implementation of label reuse, since it is too expensive
            # to predict the nodes in validation and test set during training time.
            if it == args.n_label_iters:
                loss = criterion(
                    pred[train_pred_idx],
                    subgraphs[-1].dstdata["labels"][train_pred_idx],
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                count = len(train_pred_idx)
                loss_sum += loss.item() * count
                total += count

            torch.cuda.empty_cache()

    return (
        evaluator(preds[train_idx], labels[train_idx]),
        loss_sum / total,
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

    # Due to the limitation of memory capacity, we calculate the average of logits 'eval_times' times.
    eval_times = 1

    preds_avg = torch.zeros(labels.shape[0], n_classes)
    for _ in range(eval_times):
        preds = torch.zeros(labels.shape[0], n_classes)

        for _it in range(args.n_label_iters + 1):
            preds_old = preds.clone()
            for input_nodes, output_nodes, subgraphs in dataloader:
                subgraphs = [b.to(device) for b in subgraphs]

                if args.use_labels:
                    add_soft_labels(
                        subgraphs[0],
                        F.softmax(preds_old[input_nodes].to(device), dim=-1),
                    )
                    update_hard_labels(subgraphs[0])

                pred = model(subgraphs, inference=True)
                preds[output_nodes] = pred.cpu()

                torch.cuda.empty_cache()

        preds_avg += preds

    preds_avg = preds_avg.to(device)
    preds_avg /= eval_times

    train_loss = criterion(preds_avg[train_idx], labels[train_idx]).item()
    val_loss = criterion(preds_avg[val_idx], labels[val_idx]).item()
    test_loss = criterion(preds_avg[test_idx], labels[test_idx]).item()

    return (
        evaluator(preds_avg[train_idx], labels[train_idx]),
        evaluator(preds_avg[val_idx], labels[val_idx]),
        evaluator(preds_avg[test_idx], labels[test_idx]),
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

    n_train_samples = train_idx.shape[0]
    train_batch_size = (n_train_samples + 29) // 30
    train_sampler = MultiLayerNeighborSampler(
        [10 for _ in range(args.n_layers)]
    )
    train_dataloader = DataLoader(
        graph.cpu(),
        train_idx.cpu(),
        train_sampler,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
    )

    eval_batch_size = 32768
    eval_sampler = MultiLayerNeighborSampler([15 for _ in range(args.n_layers)])

    if args.estimation_mode:
        test_idx_during_training = test_idx[
            torch.arange(start=0, end=len(test_idx), step=45)
        ]
    else:
        test_idx_during_training = test_idx

    eval_idx = torch.cat(
        [train_idx.cpu(), val_idx.cpu(), test_idx_during_training.cpu()]
    )
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

        score, loss = train(
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

        if (
            epoch == args.n_epochs
            or epoch % args.eval_every == 0
            or epoch % args.log_every == 0
        ):
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
                test_idx_during_training,
                criterion,
                evaluator_wrapper,
            )

            if val_score > best_val_score:
                best_val_score = val_score
                final_test_score = test_score
                if args.estimation_mode:
                    best_model_state_dict = {
                        k: v.to("cpu") for k, v in model.state_dict().items()
                    }

            if epoch == args.n_epochs or epoch % args.log_every == 0:
                print(
                    f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}\n"
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

    if args.estimation_mode:
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
        f"Best val score: {best_val_score}, Final test score: {final_test_score}"
    )
    print("*" * 50)

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
        ax.xaxis.set_major_locator(MultipleLocator(10))
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
        ax.xaxis.set_major_locator(MultipleLocator(10))
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
        "GAT implementation on ogbn-products",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument(
        "--cpu",
        action="store_true",
        help="CPU mode. This option overrides '--gpu'.",
    )
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    argparser.add_argument("--seed", type=int, default=0, help="seed")
    argparser.add_argument(
        "--n-runs", type=int, default=10, help="running times"
    )
    argparser.add_argument(
        "--n-epochs", type=int, default=250, help="number of epochs"
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
        "--no-attn-dst", action="store_true", help="Don't use attn_dst."
    )
    argparser.add_argument(
        "--mask-rate", type=float, default=0.5, help="mask rate"
    )
    argparser.add_argument(
        "--n-heads", type=int, default=4, help="number of heads"
    )
    argparser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate"
    )
    argparser.add_argument(
        "--n-layers", type=int, default=3, help="number of layers"
    )
    argparser.add_argument(
        "--n-hidden", type=int, default=120, help="number of hidden units"
    )
    argparser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout rate"
    )
    argparser.add_argument(
        "--input-drop", type=float, default=0.1, help="input drop rate"
    )
    argparser.add_argument(
        "--attn-dropout", type=float, default=0.0, help="attention drop rate"
    )
    argparser.add_argument(
        "--edge-drop", type=float, default=0.1, help="edge drop rate"
    )
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument(
        "--eval-every", type=int, default=2, help="log every EVAL_EVERY epochs"
    )
    argparser.add_argument(
        "--estimation-mode",
        action="store_true",
        help="Estimate the score of test set for speed during training.",
    )
    argparser.add_argument(
        "--log-every", type=int, default=2, help="log every LOG_EVERY epochs"
    )
    argparser.add_argument(
        "--plot-curves", action="store_true", help="plot learning curves"
    )
    args = argparser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % args.gpu)

    # load data & preprocess
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset)
    graph, labels = preprocess(graph, labels, train_idx)

    labels, train_idx, val_idx, test_idx = map(
        lambda x: x.to(device), (labels, train_idx, val_idx, test_idx)
    )

    # run
    val_scores, test_scores = [], []

    for i in range(1, args.n_runs + 1):
        seed(args.seed + i)
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
            "WARNING: Estimation mode is enabled. The final test score is accurate, but not accurate during training time."
        )


if __name__ == "__main__":
    main()

# Namespace(attn_dropout=0.0, cpu=False, dropout=0.5, edge_drop=0.1, estimation_mode=True, eval_every=2, gpu=1, input_drop=0.1, log_every=2, lr=0.01, mask_rate=0.5, n_epochs=250, n_heads=4, n_hidden=120, n_label_iters=0, n_layers=3, n_runs=10, no_attn_dst=False, plot_curves=True, seed=0, use_labels=False, wd=0)
# Runned 10 times
# Val scores: [0.9326348447473489, 0.9330163008926073, 0.9327619967957684, 0.932355110240826, 0.9330163008926073, 0.9327365663860845, 0.9329145792538718, 0.9322788190117742, 0.9321516669633548, 0.9329908704829235]
# Test scores: [0.8147550191112792, 0.8115680737936217, 0.8128332725586069, 0.8134062268564646, 0.8118784993477448, 0.8145462613150566, 0.8151228304665284, 0.8115274066904614, 0.8108545920615103, 0.8094583548530088]
# Average val score: 0.9326857055667167 ± 0.00030580001557474636
# Average test score: 0.8125950537054282 ± 0.001765025824381352
# Number of params: 1065127

# Namespace(attn_dropout=0.0, cpu=False, dropout=0.5, edge_drop=0.1, estimation_mode=True, eval_every=2, gpu=0, input_drop=0.1, log_every=2, lr=0.01, mask_rate=0.5, n_epochs=250, n_heads=4, n_hidden=120, n_label_iters=0, n_layers=3, n_runs=5, no_attn_dst=True, plot_curves=True, seed=0, use_labels=False, wd=0)
# Runned 10 times
# Val scores: [0.9332451745797625, 0.9330417313022913, 0.9328128576151362, 0.9323296798311421, 0.9324568318795616, 0.9327874272054523, 0.9327619967957684, 0.9328128576151362, 0.9322025277827226, 0.9329400096635557]
# Test scores: [0.8103399272781824, 0.8115870517750965, 0.8107294277551171, 0.8115771109276573, 0.8130244079434601, 0.8094628734200265, 0.8105681149125815, 0.809217063374258, 0.8108085026779287, 0.8151549122923549]
# Average val score: 0.932739109427053 ± 0.0003061065079170266
# Average test score: 0.8112469392356664 ± 0.0016644261188834386
# Number of params: 1060887
