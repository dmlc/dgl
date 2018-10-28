import random
import sys
import time

import dgl
import dgl.backend as F
import dgl.utils as utils
import igraph
import networkx as nx
import numpy as np
import torch as th
import scipy.sparse as sp

def test_bfs():
    g = dgl.DGLGraph()
    a = sp.random(n, n, 10 / n, data_rvs=lambda n: np.ones(n))
    g.from_scipy_sparse_matrix(a)

    ig = igraph.Graph(directed=True)
    ig.add_vertices(n)
    ig.add_edges(list(zip(a.row, a.col)))

    src = random.choice(range(n))

    t0 = time.time()
    layers_cpp = g.bfs([src], out=True)
    t_dgl = time.time() - t0

    t0 = time.time()
    v, delimeter, _ = ig.bfs(src)
    t_igraph = time.time() - t0

    layers_igraph = [v[i : j] for i, j in zip(delimeter[:-1], delimeter[1:])]
    toset = lambda x: set(utils.toindex(x).tousertensor().numpy())
    assert len(layers_cpp) == len(layers_igraph)
    assert all(toset(x) == set(y) for x, y in zip(layers_cpp, layers_igraph))

    return t_dgl, t_igraph

def test_dfs(example=False):
    if example: # the example in networkx documentation
        g_nx = nx.DiGraph([(0, 1), (1, 2), (2, 1)])
        g = dgl.DGLGraph()
        g.from_networkx(g_nx)
        src = [0]
    else:
        g_nx = nx.generators.trees.random_tree(n)
        g = dgl.DGLGraph()
        g.from_networkx(g_nx)
        src = [random.choice(range(g.number_of_nodes()))]

    t0 = time.time()
    ret = g.dfs_labeled_edges(src)
    t_dgl = time.time() - t0

    t0 = time.time()
    src_nx, dst_nx, type_nx = zip(*nx.dfs_labeled_edges(g_nx, src[0]))
    t_nx = time.time() - t0

    src_tuple, dst_tuple, type_tuple = list(zip(*ret))
    src_dgl = F.pack(src_tuple)
    dst_dgl = F.pack(dst_tuple)
    type_dgl = F.pack(type_tuple)

    dict_dgl = dict(zip(zip(src_dgl.numpy(), dst_dgl.numpy()), type_dgl.numpy()))
    dict_nx = dict(zip(zip(src_nx, dst_nx), type_nx))
    dict_nx = {k : v for k, v in dict_nx.items() if k[0] != k[1]}

    assert len(dict_dgl) == len(dict_nx)
    mapping = {g.FORWARD : 'forward', g.REVERSE : 'reverse', g.NONTREE : 'nontree'}
    assert all(mapping[v] == dict_nx[k] for k, v in dict_dgl.items())

    return t_nx, t_dgl

def test_topological_traversl():
    g = dgl.DGLGraph()
    a = sp.random(n, n, 10 / n, data_rvs=lambda n: np.ones(n))
    b = sp.tril(a, -1).tocoo()
    g.from_scipy_sparse_matrix(b)

    t0 = time.time()
    layers_cpp = list(g.topological_traversal())
    t_dgl = time.time() - t0

    adjmat = g.adjacency_matrix()
    def tensor_topo_traverse():
        n = g.number_of_nodes()
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

    toset = lambda x: set(utils.toindex(x).tousertensor().numpy())
    assert len(layers_cpp) == len(layers_spmv)
    assert all(toset(x) == toset(y) for x, y in zip(layers_cpp, layers_spmv))

    return t_dgl, t_spmv

if __name__ == '__main__':
    n = int(sys.argv[1])
    N = 10

    s_dgl = 0
    s_igraph = 0
    for i in range(N):
        t_dgl, t_igraph = test_bfs()
        s_dgl += t_dgl
        s_igraph += t_igraph
    print('#######')
    print('# bfs #')
    print('#######')
    print('dgl: %.5f s' % (t_dgl / N))
    print('igraph: %.5f s' % (t_igraph / N))

    test_dfs(example=True)

    s_dgl = 0
    s_nx = 0
    for i in range(N):
        t_dgl, t_nx = test_dfs()
        s_dgl += t_dgl
        s_nx += t_nx
    print('#######')
    print('# dfs #')
    print('#######')
    print('dgl: %.5f s' % (s_dgl / N))
    print('networkx: %.5f s' % (s_nx / N))

    s_dgl = 0
    s_spmv = 0
    for i in range(N):
        t_dgl, t_spmv = test_topological_traversl()
        s_dgl += t_dgl
        s_spmv += t_spmv
    print('#########################')
    print('# topological traversal #')
    print('#########################')
    print('dgl: %.5f s' % (s_dgl / N))
    print('spmv: %.5f s' % (s_spmv / N))
