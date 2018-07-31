import torch as th
import numpy as np
import networkx as nx
from dgl import DGLGraph
from dgl.cached_graph import *

def check_eq(a, b):
    assert a.shape == b.shape
    assert th.sum(a == b) == int(np.prod(list(a.shape)))

def test_basics():
    g = DGLGraph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 4)
    g.add_edge(2, 5)
    cg = create_cached_graph(g)
    u = th.tensor([0, 1, 1, 2, 2])
    v = th.tensor([1, 2, 3, 4, 5])
    check_eq(cg.get_edge_id(u, v), th.tensor([0, 1, 2, 3, 4]))
    cg.add_edges(0, 2)
    assert cg.get_edge_id(0, 2) == 5
    query = th.tensor([1, 2])
    s, d = cg.in_edges(query)
    check_eq(s, th.tensor([0, 0, 1]))
    check_eq(d, th.tensor([1, 2, 2]))
    s, d = cg.out_edges(query)
    check_eq(s, th.tensor([1, 1, 2, 2]))
    check_eq(d, th.tensor([2, 3, 4, 5]))

    print(cg._graph.get_adjacency())
    print(cg._graph.get_adjacency(eids=True))

if __name__ == '__main__':
    test_basics()
