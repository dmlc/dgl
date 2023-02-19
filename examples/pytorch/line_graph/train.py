"""
Supervised Community Detection with Hierarchical Graph Neural Networks
https://arxiv.org/abs/1705.08415

Author's implementation: https://github.com/joanbruna/GNN_community
"""

from __future__ import division

import argparse
import time
from itertools import permutations

import gnn
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim

from dgl.data import SBMMixtureDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, help="Batch size", default=1)
parser.add_argument("--gpu", type=int, help="GPU index", default=-1)
parser.add_argument("--lr", type=float, help="Learning rate", default=0.001)
parser.add_argument(
    "--n-communities", type=int, help="Number of communities", default=2
)
parser.add_argument(
    "--n-epochs", type=int, help="Number of epochs", default=100
)
parser.add_argument(
    "--n-features", type=int, help="Number of features", default=16
)
parser.add_argument("--n-graphs", type=int, help="Number of graphs", default=10)
parser.add_argument("--n-layers", type=int, help="Number of layers", default=30)
parser.add_argument(
    "--n-nodes", type=int, help="Number of nodes", default=10000
)
parser.add_argument("--optim", type=str, help="Optimizer", default="Adam")
parser.add_argument("--radius", type=int, help="Radius", default=3)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

dev = th.device("cpu") if args.gpu < 0 else th.device("cuda:%d" % args.gpu)
K = args.n_communities

training_dataset = SBMMixtureDataset(args.n_graphs, args.n_nodes, K)
training_loader = DataLoader(
    training_dataset,
    args.batch_size,
    collate_fn=training_dataset.collate_fn,
    drop_last=True,
)

ones = th.ones(args.n_nodes // K)
y_list = [
    th.cat([x * ones for x in p]).long().to(dev) for p in permutations(range(K))
]

feats = [1] + [args.n_features] * args.n_layers + [K]
model = gnn.GNN(feats, args.radius, K).to(dev)
optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)


def compute_overlap(z_list):
    ybar_list = [th.max(z, 1)[1] for z in z_list]
    overlap_list = []
    for y_bar in ybar_list:
        accuracy = max(th.sum(y_bar == y).item() for y in y_list) / args.n_nodes
        overlap = (accuracy - 1 / K) / (1 - 1 / K)
        overlap_list.append(overlap)
    return sum(overlap_list) / len(overlap_list)


def from_np(f, *args):
    def wrap(*args):
        new = [
            th.from_numpy(x) if isinstance(x, np.ndarray) else x for x in args
        ]
        return f(*new)

    return wrap


@from_np
def step(i, j, g, lg, deg_g, deg_lg, pm_pd):
    """One step of training."""
    g = g.to(dev)
    lg = lg.to(dev)
    deg_g = deg_g.to(dev).unsqueeze(1)
    deg_lg = deg_lg.to(dev).unsqueeze(1)
    pm_pd = pm_pd.to(dev)
    t0 = time.time()
    z = model(g, lg, deg_g, deg_lg, pm_pd)
    t_forward = time.time() - t0

    z_list = th.chunk(z, args.batch_size, 0)
    loss = (
        sum(min(F.cross_entropy(z, y) for y in y_list) for z in z_list)
        / args.batch_size
    )
    overlap = compute_overlap(z_list)

    optimizer.zero_grad()
    t0 = time.time()
    loss.backward()
    t_backward = time.time() - t0
    optimizer.step()

    return loss, overlap, t_forward, t_backward


@from_np
def inference(g, lg, deg_g, deg_lg, pm_pd):
    g = g.to(dev)
    lg = lg.to(dev)
    deg_g = deg_g.to(dev).unsqueeze(1)
    deg_lg = deg_lg.to(dev).unsqueeze(1)
    pm_pd = pm_pd.to(dev)

    z = model(g, lg, deg_g, deg_lg, pm_pd)

    return z


def test():
    p_list = [6, 5.5, 5, 4.5, 1.5, 1, 0.5, 0]
    q_list = [0, 0.5, 1, 1.5, 4.5, 5, 5.5, 6]
    N = 1
    overlap_list = []
    for p, q in zip(p_list, q_list):
        dataset = SBMMixtureDataset(N, args.n_nodes, K, pq=[[p, q]] * N)
        loader = DataLoader(dataset, N, collate_fn=dataset.collate_fn)
        g, lg, deg_g, deg_lg, pm_pd = next(iter(loader))
        z = inference(g, lg, deg_g, deg_lg, pm_pd)
        overlap_list.append(compute_overlap(th.chunk(z, N, 0)))
    return overlap_list


n_iterations = args.n_graphs // args.batch_size
for i in range(args.n_epochs):
    total_loss, total_overlap, s_forward, s_backward = 0, 0, 0, 0
    for j, [g, lg, deg_g, deg_lg, pm_pd] in enumerate(training_loader):
        loss, overlap, t_forward, t_backward = step(
            i, j, g, lg, deg_g, deg_lg, pm_pd
        )

        total_loss += loss
        total_overlap += overlap
        s_forward += t_forward
        s_backward += t_backward

        epoch = "0" * (len(str(args.n_epochs)) - len(str(i)))
        iteration = "0" * (len(str(n_iterations)) - len(str(j)))
        if args.verbose:
            print(
                "[epoch %s%d iteration %s%d]loss %.3f | overlap %.3f"
                % (epoch, i, iteration, j, loss, overlap)
            )

    epoch = "0" * (len(str(args.n_epochs)) - len(str(i)))
    loss = total_loss / (j + 1)
    overlap = total_overlap / (j + 1)
    t_forward = s_forward / (j + 1)
    t_backward = s_backward / (j + 1)
    print(
        "[epoch %s%d]loss %.3f | overlap %.3f | forward time %.3fs | backward time %.3fs"
        % (epoch, i, loss, overlap, t_forward, t_backward)
    )

    overlap_list = test()
    overlap_str = " - ".join(["%.3f" % overlap for overlap in overlap_list])
    print("[epoch %s%d]overlap: %s" % (epoch, i, overlap_str))
