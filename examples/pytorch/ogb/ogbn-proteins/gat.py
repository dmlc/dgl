#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random
import time

import dgl
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
from utils import BatchSampler, DataLoaderWrapper

device = None
dataset = "ogbn-proteins"
n_node_feats, n_edge_feats, n_classes = 0, 8, 112


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
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    graph.ndata["labels"] = labels

    return graph, labels, train_idx, val_idx, test_idx, evaluator


def preprocess(graph, labels):
    global n_node_feats

    # The sum of the weights of adjacent edges is used as node features.
    graph.update_all(fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat"))
    n_node_feats = graph.ndata["feat"].shape[-1]

    graph.create_formats_()

    return graph, labels


def gen_model(args):
    model = GAT(
        n_node_feats,
        n_edge_feats,
        n_classes,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_hidden=args.n_hidden,
        edge_emb=16,
        activation=F.relu,
        dropout=args.dropout,
        input_drop=args.input_drop,
        attn_drop=args.attn_drop,
        edge_drop=args.edge_drop,
        use_attn_dst=not args.no_attn_dst,
    )

    return model


def train(_args, model, dataloader, _labels, _train_idx, criterion, optimizer, _evaluator):
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

        torch.cuda.empty_cache()

    return loss_sum / total


@torch.no_grad()
def evaluate(_args, model, dataloader, labels, train_idx, val_idx, test_idx, criterion, evaluator):
    model.eval()

    preds = torch.zeros(labels.shape).to(device)

    # Due to the limitation of memory capacity, we calculate the average of logits 'eval_times' times.
    eval_times = 1

    for _ in range(eval_times):
        for _input_nodes, output_nodes, subgraphs in dataloader:
            subgraphs = [b.to(device) for b in subgraphs]

            pred = model(subgraphs)
            preds[output_nodes] += pred

            torch.cuda.empty_cache()

    preds /= eval_times

    train_loss = criterion(preds[train_idx], labels[train_idx].float()).item()
    val_loss = criterion(preds[val_idx], labels[val_idx].float()).item()
    test_loss = criterion(preds[test_idx], labels[test_idx].float()).item()

    return (
        evaluator(preds[train_idx], labels[train_idx]),
        evaluator(preds[val_idx], labels[val_idx]),
        evaluator(preds[test_idx], labels[test_idx]),
        train_loss,
        val_loss,
        test_loss,
    )


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running):
    evaluator_wrapper = lambda pred, labels: evaluator.eval({"y_pred": pred, "y_true": labels})["rocauc"]

    train_batch_size = (len(train_idx) + 9) // 10
    # batch_size = len(train_idx)
    train_sampler = MultiLayerNeighborSampler([16 for _ in range(args.n_layers)])
    # sampler = MultiLayerFullNeighborSampler(args.n_layers)
    train_dataloader = DataLoaderWrapper(
        NodeDataLoader(
            graph.cpu(),
            train_idx.cpu(),
            train_sampler,
            batch_sampler=BatchSampler(len(train_idx), batch_size=train_batch_size),
            num_workers=4,
        )
    )

    eval_sampler = MultiLayerNeighborSampler([60 for _ in range(args.n_layers)])
    # sampler = MultiLayerFullNeighborSampler(args.n_layers)
    eval_dataloader = DataLoaderWrapper(
        NodeDataLoader(
            graph.cpu(),
            torch.cat([train_idx.cpu(), val_idx.cpu(), test_idx.cpu()]),
            eval_sampler,
            batch_sampler=BatchSampler(graph.number_of_nodes(), batch_size=32768),
            num_workers=4,
        )
    )

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

        loss = train(args, model, train_dataloader, labels, train_idx, criterion, optimizer, evaluator_wrapper)

        toc = time.time()
        total_time += toc - tic

        if epoch == args.n_epochs or epoch % args.eval_every == 0 or epoch % args.log_every == 0:
            train_score, val_score, test_score, train_loss, val_loss, test_loss = evaluate(
                args, model, eval_dataloader, labels, train_idx, val_idx, test_idx, criterion, evaluator_wrapper
            )

            if val_score > best_val_score:
                best_val_score = val_score
                final_test_score = test_score

            if epoch % args.log_every == 0:
                print(
                    f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}s"
                )
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
    print(f"Best val score: {best_val_score}, Final test score: {final_test_score}")
    print("*" * 50)

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
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    argparser.add_argument("--seed", type=int, default=0, help="random seed")
    argparser.add_argument("--n-runs", type=int, default=10, help="running times")
    argparser.add_argument("--n-epochs", type=int, default=1200, help="number of epochs")
    argparser.add_argument("--no-attn-dst", action="store_true", help="Don't use attn_dst.")
    argparser.add_argument("--n-heads", type=int, default=6, help="number of heads")
    argparser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=6, help="number of layers")
    argparser.add_argument("--n-hidden", type=int, default=80, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.25, help="dropout rate")
    argparser.add_argument("--input-drop", type=float, default=0.1, help="input drop rate")
    argparser.add_argument("--attn-drop", type=float, default=0.0, help="attention dropout rate")
    argparser.add_argument("--edge-drop", type=float, default=0.1, help="edge drop rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--eval-every", type=int, default=5, help="evaluate every EVAL_EVERY epochs")
    argparser.add_argument("--log-every", type=int, default=5, help="log every LOG_EVERY epochs")
    argparser.add_argument("--plot-curves", action="store_true", help="plot learning curves")
    args = argparser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    seed(args.seed)

    # load data & preprocess
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset)
    graph, labels = preprocess(graph, labels)

    labels, train_idx, val_idx, test_idx = map(lambda x: x.to(device), (labels, train_idx, val_idx, test_idx))

    # run
    val_scores, test_scores = [], []

    for i in range(1, args.n_runs + 1):
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

# Namespace(attn_drop=0.0, cpu=False, dropout=0.25, edge_drop=0.1, eval_every=5, gpu=3, input_drop=0.1, log_every=5, lr=0.01, n_epochs=1200, n_heads=6, n_hidden=80, n_layers=6, n_runs=10, no_attn_dst=False, plot_curves=True, seed=0, wd=0)
# Runned 10 times
# Val scores: [0.9191360923559534, 0.9190902739433363, 0.91955548684813, 0.91938196250152, 0.9188466895131054, 0.9197077629866909, 0.9197391599757976, 0.9195898788031738, 0.9193022233033481, 0.9193399302341642]
# Test scores: [0.8667826604519867, 0.8687609429765975, 0.8672772211837847, 0.8681650855671987, 0.8699117519793396, 0.8690210391385448, 0.8728497958960492, 0.8685245877500837, 0.864323193259458, 0.8668325831720998]
# Average val score: 0.9193689460465219 ± 0.0002730491100383192
# Average test score: 0.8682448861375143 ± 0.0021303924285955614
# Number of params: 2475232

# Namespace(attn_drop=0.0, cpu=False, dropout=0.25, edge_drop=0.1, eval_every=5, gpu=3, input_drop=0.1, log_every=5, lr=0.01, n_epochs=1200, n_heads=6, n_hidden=80, n_layers=6, n_runs=10, no_attn_dst=True, plot_curves=True, seed=0, wd=0)
# Runned 10 times
# Val scores: [0.9189444999090346, 0.9195528736759115, 0.9182374839397544, 0.9195157117554997, 0.920224160572496, 0.9178686384904079, 0.9187506919340453, 0.9201368191117556, 0.9194343717555682, 0.9195504427450948]
# Test scores: [0.8700498833734864, 0.8668706856410779, 0.865478472559448, 0.8684981160429261, 0.8678271665110211, 0.868009225355837, 0.8698801263531729, 0.86619033217202, 0.86611749081188, 0.8682289996146542]
# Average val score: 0.9192215693889567 ± 0.0007273194066010714
# Average test score: 0.8677150498435523 ± 0.0014733512146035438
# Number of params: 2460352
