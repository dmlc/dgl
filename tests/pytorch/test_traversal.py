import random
import sys
import time

import dgl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
import utils as U

import itertools

np.random.seed(42)

def toset(x):
    return set(x.tolist())

def test_bfs_nodes(n=1000):
    g_nx = nx.random_tree(n)
    g = dgl.DGLGraph()
    g.from_networkx(g_nx)

    src = random.choice(range(n))

    edges = nx.bfs_edges(g_nx, src)
    layers_nx = [set([src])]
    edges_nx = []
    frontier = set()
    edge_frontier = set()
    for u, v in edges:
        if u in layers_nx[-1]:
            frontier.add(v)
            edge_frontier.add(g.edge_id(u, v))
        else:
            layers_nx.append(frontier)
            edges_nx.append(edge_frontier)
            frontier = set([v])
            edge_frontier = set([g.edge_id(u, v)])
    layers_nx.append(frontier)
    edges_nx.append(edge_frontier)

    layers_dgl = dgl.bfs_nodes_generator(g, src)

    assert len(layers_dgl) == len(layers_nx)
    assert all(toset(x) == y for x, y in zip(layers_dgl, layers_nx))

    edges_dgl = dgl.bfs_edges_generator(g, src)

    assert len(edges_dgl) == len(edges_nx)
    assert all(toset(x) == y for x, y in zip(edges_dgl, edges_nx))

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

    assert len(layers_dgl) == len(layers_spmv)
    assert all(toset(x) == toset(y) for x, y in zip(layers_dgl, layers_spmv))

DFS_LABEL_NAMES = ['forward', 'reverse', 'nontree']
def test_dfs_labeled_edges(n=1000, example=False):
    dgl_g = dgl.DGLGraph()
    dgl_g.add_nodes(6)
    dgl_g.add_edges([0, 1, 0, 3, 3], [1, 2, 2, 4, 5])
    dgl_edges, dgl_labels = dgl.dfs_labeled_edges_generator(
            dgl_g, [0, 3], has_reverse_edge=True, has_nontree_edge=True)
    dgl_edges = [toset(t) for t in dgl_edges]
    dgl_labels = [toset(t) for t in dgl_labels]

    g1_solutions = [
            # edges           labels
            [[0, 1, 1, 0, 2], [0, 0, 1, 1, 2]],
            [[2, 2, 0, 1, 0], [0, 1, 0, 2, 1]],
    ]
    g2_solutions = [
            # edges        labels
            [[3, 3, 4, 4], [0, 1, 0, 1]],
            [[4, 4, 3, 3], [0, 1, 0, 1]],
    ]

    def combine_frontiers(sol):
        es, ls = zip(*sol)
        es = [set(i for i in t if i is not None)
              for t in itertools.zip_longest(*es)]
        ls = [set(i for i in t if i is not None)
              for t in itertools.zip_longest(*ls)]
        return es, ls

    for sol_set in itertools.product(g1_solutions, g2_solutions):
        es, ls = combine_frontiers(sol_set)
        if es == dgl_edges and ls == dgl_labels:
            break
    else:
        assert False


if __name__ == '__main__':
    test_bfs_nodes()
    test_topological_nodes()
    test_dfs_labeled_edges()
