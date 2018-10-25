import random
import sys
import time

import dgl
import dgl.utils as utils
import igraph
import numpy as np
import scipy.sparse as sp

g = dgl.DGLGraph()
n = int(sys.argv[1])
a = sp.random(n, n, 10 / n, data_rvs=lambda n: np.ones(n))
g.from_scipy_sparse_matrix(a)

ig = igraph.Graph(directed=True)
ig.add_vertices(n)
ig.add_edges(list(zip(a.row, a.col)))

src = random.choice(range(n))

t0 = time.time()
layers_cpp = g._graph.bfs([src], out=True)
t_cpp = time.time() - t0

t0 = time.time()
v, delimeter, _ = ig.bfs(src)
t_igraph = time.time() - t0

print(t_cpp, t_igraph)
layers_igraph = [v[i : j] for i, j in zip(delimeter[:-1], delimeter[1:])]
toset = lambda x: set(utils.toindex(x).tousertensor().numpy())
assert len(layers_cpp) == len(layers_igraph)
assert all(toset(x) == set(y) for x, y in zip(layers_cpp, layers_igraph))
