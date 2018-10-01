"""
ipython3 train.py -- --gpu -1 --n-classes 2 --n-iterations 1000 --n-layers 30 --n-nodes 1000 --n-features 2 --radius 3
"""

import argparse
from itertools import permutations
import networkx as nx
import torch as th
import torch.nn.functional as F
import torch.optim as optim
import gnn
import sbm

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int)
parser.add_argument('--gpu', type=int)
parser.add_argument('--n-classes', type=int)
parser.add_argument('--n-features', type=int)
parser.add_argument('--n-graphs', type=int)
parser.add_argument('--n-iterations', type=int)
parser.add_argument('--n-layers', type=int)
parser.add_argument('--n-nodes', type=int)
parser.add_argument('--radius', type=int)
args = parser.parse_args()

dev = th.device('cpu') if args.gpu < 0 else th.device('cuda:%d' % args.gpu)

ssbm = sbm.SSBM(args.n_nodes, args.n_classes, 1, 1)
gg = []
for i in range(args.n_graphs):
    ssbm.generate()
    gg.append(ssbm.graph)

assert args.n_nodes % args.n_classes == 0
ones = th.ones(int(args.n_nodes / args.n_classes))
yy = [th.cat([x * ones for x in p]).long().to(dev)
      for p in permutations(range(args.n_classes))]

feats = [1] + [args.n_features] * args.n_layers + [args.n_classes]
model = gnn.GNN(g, feats, args.radius, args.n_classes).to(dev)
opt = optim.Adamax(model.parameters(), lr=0.04)

for i in range(args.n_iterations):
    y_bar = model()
    loss = min(F.cross_entropy(y_bar, y) for y in yy)
    opt.zero_grad()
    loss.backward()
    opt.step()

    print('[iteration %d]loss %f' % (i, loss))
