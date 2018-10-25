import random
import sys
import time

import dgl
import dgl.backend as F
import dgl.ndarray as nd
import dgl.utils as utils
import igraph
import networkx as nx
import numpy as np
import torch as th
import scipy.sparse as sp

g = dgl.DGLGraph()
n = int(sys.argv[1])
a = sp.random(n, n, 10 / n, data_rvs=lambda n: np.ones(n))
b = sp.tril(a, -1).tocoo()
g.from_scipy_sparse_matrix(b)

t0 = time.time()
layers_cpp = list(g._graph.topological_traversal(out=True, step=False))
t_cpp = time.time() - t0

n = g.number_of_nodes()
adjmat = g._graph.adjacency_matrix().get(nd.cpu())
def tensor_topo_traverse():
    mask = th.ones((n, 1))
    degree = th.spmm(adjmat, mask)
    while th.sum(mask) != 0.:
        v = (degree == 0.).float()
        v = v * mask
        mask = mask - v
        frontier = th.squeeze(th.squeeze(v).nonzero(), 1)
        yield frontier
        degree -= th.spmm(adjmat, v)

t0 = time.time()
layers_spmv = list(tensor_topo_traverse())
t_spmv = time.time() - t0

print(t_cpp, t_spmv)
toset = lambda x: set(utils.toindex(x).tousertensor().numpy())
assert len(layers_cpp) == len(layers_spmv)
assert all(toset(x) == toset(y) for x, y in zip(layers_cpp, layers_spmv))
