import argparse
import time

import dgl
import dgl.backend as F
import igraph
import networkx as nx
import numpy as np
import scipy.sparse as sp

parser = argparse.ArgumentParser()
parser.add_argument('--bt', action='store_true', help='BackTracking')
parser.add_argument('--n-nodes', type=int, help='Number of NODES')
args = parser.parse_args()

n = args.n_nodes
a = sp.random(n, n, 1 / n, data_rvs=lambda n: np.ones(n))
b = sp.triu(a, 1) + sp.triu(a, 1).transpose()
g = dgl.DGLGraph()
g.from_scipy_sparse_matrix(b)

N = 10

t0 = time.time()
for i in range(N):
    lg_sparse = g.line_graph(args.bt)
    g._graph._cache.clear()
t = (time.time() - t0) / N
print('dgl.DGLGraph.line_graph: %f' % t)

t0 = time.time()
for i in range(N):
    lg = g._line_graph(args.bt)
    g._graph._cache.clear()
t = (time.time() - t0) / N
print('dgl.DGLGraph._line_graph: %f' % t)

g = igraph.Graph()
g.add_vertices(n)
g.add_edges(list(zip(a.row.tolist(), a.col.tolist())))

'''
t0 = time.time()
for i in range(N):
    lg = g.linegraph()
t = (time.time() - t0) / N
print('igraph.Graph._line_graph: %f' % t)
'''

t = 0
for i in range(N):
    g = igraph.Graph()
    g.add_vertices(n)
    g.add_edges(list(zip(a.row.tolist(), a.col.tolist())))
    t0 = time.time()
    lg = g.linegraph()
    t += time.time() - t0
t /= N
print('igraph.Graph.linegraph: %f' % t)
