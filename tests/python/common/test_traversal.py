import itertools
import random
import sys
import time
import unittest

import backend as F

import dgl
import networkx as nx
import numpy as np
import scipy.sparse as sp
from utils import parametrize_idtype

np.random.seed(42)


def toset(x):
    # F.zerocopy_to_numpy may return a int
    return set(F.zerocopy_to_numpy(x).tolist())


@parametrize_idtype
def test_bfs(idtype, n=100):
    def _bfs_nx(g_nx, src):
        edges = nx.bfs_edges(g_nx, src)
        layers_nx = [set([src])]
        edges_nx = []
        frontier = set()
        edge_frontier = set()
        for u, v in edges:
            if u in layers_nx[-1]:
                frontier.add(v)
                edge_frontier.add(g.edge_ids(int(u), int(v)))
            else:
                layers_nx.append(frontier)
                edges_nx.append(edge_frontier)
                frontier = set([v])
                edge_frontier = set([g.edge_ids(u, v)])
        # avoids empty successors
        if len(frontier) > 0 and len(edge_frontier) > 0:
            layers_nx.append(frontier)
            edges_nx.append(edge_frontier)
        return layers_nx, edges_nx

    a = sp.random(n, n, 3 / n, data_rvs=lambda n: np.ones(n))
    g = dgl.from_scipy(a).astype(idtype)

    g_nx = g.to_networkx()
    src = random.choice(range(n))
    layers_nx, _ = _bfs_nx(g_nx, src)
    layers_dgl = dgl.bfs_nodes_generator(g, src)
    assert len(layers_dgl) == len(layers_nx)
    assert all(toset(x) == y for x, y in zip(layers_dgl, layers_nx))

    g_nx = nx.random_labeled_tree(n, seed=42)
    g = dgl.from_networkx(g_nx).astype(idtype)
    src = 0
    _, edges_nx = _bfs_nx(g_nx, src)
    edges_dgl = dgl.bfs_edges_generator(g, src)
    assert len(edges_dgl) == len(edges_nx)
    assert all(toset(x) == y for x, y in zip(edges_dgl, edges_nx))


@parametrize_idtype
def test_topological_nodes(idtype, n=100):
    a = sp.random(n, n, 3 / n, data_rvs=lambda n: np.ones(n))
    b = sp.tril(a, -1).tocoo()
    g = dgl.from_scipy(b).astype(idtype)

    layers_dgl = dgl.topological_nodes_generator(g)

    adjmat = g.adj_external(transpose=True)

    def tensor_topo_traverse():
        n = g.num_nodes()
        mask = F.copy_to(F.ones((n, 1)), F.cpu())
        degree = F.spmm(adjmat, mask)
        while F.reduce_sum(mask) != 0.0:
            v = F.astype((degree == 0.0), F.float32)
            v = v * mask
            mask = mask - v
            frontier = F.copy_to(F.nonzero_1d(F.squeeze(v, 1)), F.cpu())
            yield frontier
            degree -= F.spmm(adjmat, v)

    layers_spmv = list(tensor_topo_traverse())

    assert len(layers_dgl) == len(layers_spmv)
    assert all(toset(x) == toset(y) for x, y in zip(layers_dgl, layers_spmv))


DFS_LABEL_NAMES = ["forward", "reverse", "nontree"]


@parametrize_idtype
def test_dfs_labeled_edges(idtype, example=False):
    dgl_g = dgl.graph([]).astype(idtype)
    dgl_g.add_nodes(6)
    dgl_g.add_edges([0, 1, 0, 3, 3], [1, 2, 2, 4, 5])
    dgl_edges, dgl_labels = dgl.dfs_labeled_edges_generator(
        dgl_g, [0, 3], has_reverse_edge=True, has_nontree_edge=True
    )
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
        es = [
            set(i for i in t if i is not None)
            for t in itertools.zip_longest(*es)
        ]
        ls = [
            set(i for i in t if i is not None)
            for t in itertools.zip_longest(*ls)
        ]
        return es, ls

    for sol_set in itertools.product(g1_solutions, g2_solutions):
        es, ls = combine_frontiers(sol_set)
        if es == dgl_edges and ls == dgl_labels:
            break
    else:
        assert False


if __name__ == "__main__":
    test_bfs(idtype="int32")
    test_topological_nodes(idtype="int32")
    test_dfs_labeled_edges(idtype="int32")
