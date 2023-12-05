import math
import unittest

import backend as F

import dgl
from utils import parametrize_idtype


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@parametrize_idtype
def test_node_homophily(idtype):
    # IfChangeThenChange: python/dgl/homophily.py
    # Update the docstring example.
    device = F.ctx()
    graph = dgl.graph(
        ([1, 2, 0, 4], [0, 1, 2, 3]), idtype=idtype, device=device
    )
    y = F.tensor([0, 0, 0, 0, 1])
    assert math.isclose(dgl.node_homophily(graph, y), 0.6000000238418579)


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@parametrize_idtype
def test_edge_homophily(idtype):
    # IfChangeThenChange: python/dgl/homophily.py
    # Update the docstring example.
    device = F.ctx()
    graph = dgl.graph(
        ([1, 2, 0, 4], [0, 1, 2, 3]), idtype=idtype, device=device
    )
    y = F.tensor([0, 0, 0, 0, 1])
    assert math.isclose(dgl.edge_homophily(graph, y), 0.75)


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@parametrize_idtype
def test_linkx_homophily(idtype):
    # IfChangeThenChange: python/dgl/homophily.py
    # Update the docstring example.
    device = F.ctx()
    graph = dgl.graph(([0, 1, 2, 3], [1, 2, 0, 4]), device=device)
    y = F.tensor([0, 0, 0, 0, 1])
    assert math.isclose(dgl.linkx_homophily(graph, y), 0.19999998807907104)

    y = F.tensor([0, 1, 2, 3, 4])
    assert math.isclose(dgl.linkx_homophily(graph, y), 0.0000000000000000)


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@parametrize_idtype
def test_adjusted_homophily(idtype):
    # IfChangeThenChange: python/dgl/homophily.py
    # Update the docstring example.
    device = F.ctx()
    graph = dgl.graph(
        ([1, 2, 0, 4], [0, 1, 2, 3]), idtype=idtype, device=device
    )
    y = F.tensor([0, 0, 0, 0, 1])
    assert math.isclose(dgl.adjusted_homophily(graph, y), -0.1428571492433548)
