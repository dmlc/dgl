import torch as th
import numpy as np
import networkx as nx
from dgl import DGLGraph
from dgl.cached_graph import *
from dgl.utils import Index

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
    g.add_edge(0, 2)
    cg = create_cached_graph(g)
    u = Index(th.tensor([0, 0, 1, 1, 2, 2]))
    v = Index(th.tensor([1, 2, 2, 3, 4, 5]))
    check_eq(cg.get_edge_id(u, v).totensor(), th.tensor([0, 5, 1, 2, 3, 4]))
    query = Index(th.tensor([1, 2]))
    s, d = cg.in_edges(query)
    check_eq(s.totensor(), th.tensor([0, 0, 1]))
    check_eq(d.totensor(), th.tensor([1, 2, 2]))
    s, d = cg.out_edges(query)
    check_eq(s.totensor(), th.tensor([1, 1, 2, 2]))
    check_eq(d.totensor(), th.tensor([2, 3, 4, 5]))

if __name__ == '__main__':
    test_basics()
