import random
import sys
import time

import dgl
import dgl.backend as F
import dgl.utils as utils
import networkx as nx
import numpy as np
import scipy.sparse as sp

def test_bfs(n=1000):
    g = dgl.DGLGraph()
    a = sp.random(n, n, 10 / n, data_rvs=lambda n: np.ones(n))
    g.from_scipy_sparse_matrix(a)
    g_nx = g.to_networkx()

    src = random.choice(range(n))

    layers_dgl = g.bfs([src])

    edges = nx.bfs_edges(g_nx, src)
    layers_nx = [set([src])]
    frontier = set()
    for u, v in edges:
        if u in layers_nx[-1]:
            frontier.add(v)
        else:
            layers_nx.append(frontier)
            frontier = set([v])
    layers_nx.append(frontier)

    toset = lambda x: set(utils.toindex(x).tousertensor().numpy())
    assert len(layers_dgl) == len(layers_nx)
    assert all(toset(x) == y for x, y in zip(layers_dgl, layers_nx))

def test_dfs(n=1000, example=False):
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

    ret = g.dfs_labeled_edges(src)
    src_tuple, dst_tuple, type_tuple = list(zip(*ret))
    src_dgl = F.pack(src_tuple)
    dst_dgl = F.pack(dst_tuple)
    type_dgl = F.pack(type_tuple)
    dict_dgl = dict(zip(zip(src_dgl.numpy(), dst_dgl.numpy()), type_dgl.numpy()))

    src_nx, dst_nx, type_nx = zip(*nx.dfs_labeled_edges(g_nx, src[0]))
    dict_nx = dict(zip(zip(src_nx, dst_nx), type_nx))
    dict_nx = {k : v for k, v in dict_nx.items() if k[0] != k[1]}

    assert len(dict_dgl) == len(dict_nx)
    mapping = {g.FORWARD : 'forward', g.REVERSE : 'reverse', g.NONTREE : 'nontree'}
    assert all(mapping[v] == dict_nx[k] for k, v in dict_dgl.items())

def test_topological_traversl(n=1000):
    g = dgl.DGLGraph()
    a = sp.random(n, n, 10 / n, data_rvs=lambda n: np.ones(n))
    b = sp.tril(a, -1).tocoo()
    g.from_scipy_sparse_matrix(b)

    layers_dgl = list(g.topological_traversal())

    adjmat = g.adjacency_matrix()
    def tensor_topo_traverse():
        n = g.number_of_nodes()
        mask = F.ones((n, 1))
        degree = F.spmm(adjmat, mask)
        while F.sum(mask) != 0.:
            v = (degree == 0.).float()
            v = v * mask
            mask = mask - v
            frontier = F.squeeze(F.squeeze(v).nonzero(), 1)
            yield frontier
            degree -= F.spmm(adjmat, v)

    layers_spmv = list(tensor_topo_traverse())

    toset = lambda x: set(utils.toindex(x).tousertensor().numpy())
    assert len(layers_dgl) == len(layers_spmv)
    assert all(toset(x) == toset(y) for x, y in zip(layers_dgl, layers_spmv))

if __name__ == '__main__':
    test_bfs()
    test_dfs()
    test_topological_traversl()
