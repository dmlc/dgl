from __future__ import division

import argparse
from itertools import permutations

import networkx as nx
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import dgl
from dgl.data import SBMMixture
import gnn
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int)
parser.add_argument('--gpu', type=int)
parser.add_argument('--n-communities', type=int)
parser.add_argument('--n-features', type=int)
parser.add_argument('--n-graphs', type=int)
parser.add_argument('--n-iterations', type=int)
parser.add_argument('--n-layers', type=int)
parser.add_argument('--n-nodes', type=int)
parser.add_argument('--model-path', type=str)
parser.add_argument('--radius', type=int)
args = parser.parse_args()

dev = th.device('cpu') if args.gpu < 0 else th.device('cuda:%d' % args.gpu)

dataset = SBMMixture(args.n_graphs, args.n_nodes, args.n_communities)
loader = utils.cycle(DataLoader(dataset, args.batch_size,
                     shuffle=True, collate_fn=dataset.collate_fn, drop_last=True))

ones = th.ones(args.n_nodes // args.n_communities)
y_list = [th.cat([th.cat([x * ones for x in p])] * args.batch_size).long().to(dev)
      for p in permutations(range(args.n_communities))]

feats = [1] + [args.n_features] * args.n_layers + [args.n_communities]
model = gnn.GNN(feats, args.radius, args.n_communities).to(dev)
opt = optim.Adamax(model.parameters(), lr=0.04)

for i in range(args.n_iterations):
    g, lg, deg_g, deg_lg, eid2nid = next(loader)
    deg_g = deg_g.to(dev)
    deg_lg = deg_lg.to(dev)
    eid2nid = eid2nid.to(dev)
    y_bar = model(g, lg, deg_g, deg_lg, eid2nid)
    loss = min(F.cross_entropy(y_bar, y) for y in y_list)
    opt.zero_grad()
    loss.backward()
    opt.step()

    placeholder = '0' * (len(str(args.n_iterations)) - len(str(i)))
    print('[iteration %s%d]loss %f' % (placeholder, i, loss))

th.save(model.state_dict(), args.model_path)
