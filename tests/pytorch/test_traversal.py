import random
import sys
import time

import dgl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
import utils as U

np.random.seed(42)

def test_bfs_nodes(n=1000):
    g = dgl.DGLGraph()
    a = sp.random(n, n, 10 / n, data_rvs=lambda n: np.ones(n))
    g.from_scipy_sparse_matrix(a)
    g_nx = g.to_networkx()

    src = random.choice(range(n))

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

    layers_dgl = dgl.bfs_nodes_generator(g, src)

    toset = lambda x: set(x.tolist())
    assert len(layers_dgl) == len(layers_nx)
    assert all(toset(x) == y for x, y in zip(layers_dgl, layers_nx))

def test_topological_nodes(n=1000):
    g = dgl.DGLGraph()
    a = sp.random(n, n, 10 / n, data_rvs=lambda n: np.ones(n))
    b = sp.tril(a, -1).tocoo()
    g.from_scipy_sparse_matrix(b)

    layers_dgl = dgl.topological_nodes_generator(g)

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

    layers_spmv = list(tensor_topo_traverse())

    toset = lambda x: set(x.tolist())
    assert len(layers_dgl) == len(layers_spmv)
    assert all(toset(x) == toset(y) for x, y in zip(layers_dgl, layers_spmv))

DFS_LABEL_NAMES = ['forward', 'reverse', 'nontree']
def test_dfs_labeled_edges(n=1000, example=False):
    nx_g = nx.DiGraph()
    nx_g.add_edges_from([(0, 1), (1, 2), (0, 2)])
    nx_rst = list(nx.dfs_labeled_edges(nx_g, 0))[1:-1]  # the first and the last are not edges

    dgl_g = dgl.DGLGraph()
    dgl_g.add_nodes(3)
    dgl_g.add_edge(0, 1)
    dgl_g.add_edge(1, 2)
    dgl_g.add_edge(0, 2)
    dgl_edges, dgl_labels = dgl.dfs_labeled_edges_generator(
            dgl_g, 0, has_reverse_edge=True, has_nontree_edge=True)
    dgl_edges = [dgl_g.find_edges(e) for e in dgl_edges]
    dgl_u = [int(u) for u, v in dgl_edges]
    dgl_v = [int(v) for u, v in dgl_edges]
    dgl_labels = [DFS_LABEL_NAMES[l] for l in dgl_labels]
    dgl_rst = list(zip(dgl_u, dgl_v, dgl_labels))
    assert nx_rst == dgl_rst

if __name__ == '__main__':
    test_bfs_nodes()
    test_topological_nodes()
    test_dfs_labeled_edges()
