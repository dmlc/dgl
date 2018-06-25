"""
ipython3 test.py -- --features 1 16 16 --gpu -1 --n-classes 5 --n-iterations 10 --n-nodes 10 --order 3 --radius 3
"""


import argparse
import networkx as nx
import torch as th
import torch.nn as nn
import torch.optim as optim
import gnn


parser = argparse.ArgumentParser()
parser.add_argument('--features', nargs='+', type=int)
parser.add_argument('--gpu', type=int)
parser.add_argument('--n-classes', type=int)
parser.add_argument('--n-iterations', type=int)
parser.add_argument('--n-nodes', type=int)
parser.add_argument('--order', type=int)
parser.add_argument('--radius', type=int)
args = parser.parse_args()


if args.gpu < 0:
    cuda = False
else:
    cuda = True
    th.cuda.set_device(args.gpu)


g = nx.barabasi_albert_graph(args.n_nodes, 1).to_directed() # TODO SBM
y = th.multinomial(th.ones(args.n_classes), args.n_nodes, replacement=True)


network = gnn.GNN(args.features, args.order, args.radius, args.n_classes)
if cuda:
    network.cuda()
ce = nn.CrossEntropyLoss()
adam = optim.Adam(network.parameters())


for i in range(args.n_iterations):
    y_bar = network(g)
    loss = ce(y_bar, y)
    adam.zero_grad()
    loss.backward()
    adam.step()

    print('[iteration %d]loss %f' % (i, loss))
