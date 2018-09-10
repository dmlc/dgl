"""
ipython3 test.py -- --features 1 16 16 --gpu -1 --n-classes 5 --n-iterations 10 --n-nodes 10 --radius 3
"""

import argparse
import networkx as nx
import torch as th
import torch.nn.functional as F
import torch.optim as optim
import gnn

parser = argparse.ArgumentParser()
parser.add_argument('--features', nargs='+', type=int)
parser.add_argument('--gpu', type=int)
parser.add_argument('--n-classes', type=int)
parser.add_argument('--n-iterations', type=int)
parser.add_argument('--n-nodes', type=int)
parser.add_argument('--radius', type=int)
args = parser.parse_args()

if args.gpu < 0:
    cuda = False
else:
    cuda = True
    th.cuda.set_device(args.gpu)

g = nx.barabasi_albert_graph(args.n_nodes, 1).to_directed() # TODO SBM
y = th.multinomial(th.ones(args.n_classes), args.n_nodes, replacement=True)
model = gnn.GNN(g, args.features, args.radius, args.n_classes)
if cuda:
    model.cuda()
opt = optim.Adam(model.parameters())

for i in range(args.n_iterations):
    y_bar = model()
    loss = F.cross_entropy(y_bar, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    print('[iteration %d]loss %f' % (i, loss))
