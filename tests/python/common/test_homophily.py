import unittest

import backend as F

import dgl
from test_utils import parametrize_idtype


@parametrize_idtype
def test_node_homophily(idtype):
    # IfChangeThenChange: python/dgl/homophily.py
    # Update the docstring example.
    device = F.ctx()
    graph = dgl.graph(
        ([1, 2, 0, 4], [0, 1, 2, 3]), idtype=idtype, device=device
    )
    y = F.tensor([0, 0, 0, 0, 1])
    assert dgl.node_homophily(graph, y) == 0.6000000238418579


@parametrize_idtype
def test_edge_homophily(idtype):
    # IfChangeThenChange: python/dgl/homophily.py
    # Update the docstring example.
    device = F.ctx()
    graph = dgl.graph(
        ([1, 2, 0, 4], [0, 1, 2, 3]), idtype=idtype, device=device
    )
    y = F.tensor([0, 0, 0, 0, 1])
    assert dgl.edge_homophily(graph, y) == 0.75
