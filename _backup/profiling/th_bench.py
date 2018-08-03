import argparse
import cProfile
import networkx as nx
import torch as th
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()


if args.gpu < 0:
    cuda = False
    device = th.device('cpu')
else:
    cuda = True
    th.cuda.set_device(args.gpu)
    device = th.device(args.gpu)


g = nx.read_graphml('pgp.xml')
n = g.number_of_nodes()


deg = g.out_degree()
for (src, trg), attrs in g.edges.items():
    attrs['weight'] = 1.0 / deg[src]
adj = utils.sparse_sp2th(nx.adj_matrix(g))
if cuda:
    adj = adj.cuda()


def pagerank(alpha, tol, max_iter):
    pr = th.full((n, 1), 1 / n, device=device)
    for i in range(max_iter):
        next_pr = (1 - alpha) / n + alpha * th.mm(adj, pr)
        if th.sum(th.abs(next_pr - pr)) < tol * n:
            break
    return next_pr


for i in range(10):
    pagerank(alpha=0.85, tol=1e-3, max_iter=10000000)


print("Profiling PageRank")
print("==================")
print()


# cProfile.run("for i in range(10): pagerank(alpha=0.85, tol=1e-3, max_iter=10000000)", sort="cumulative")


import time
t0 = time.time()
for i in range(10):
    pagerank(alpha=0.85, tol=1e-3, max_iter=10000000)
print((time.time() - t0) / 10)
