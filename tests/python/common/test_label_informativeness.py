import math
import unittest

import backend as F

import dgl
from utils import parametrize_idtype


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@parametrize_idtype
def test_edge_label_informativeness(idtype):
    # IfChangeThenChange: python/dgl/label_informativeness.py
    # Update the docstring example.
    device = F.ctx()
    graph = dgl.graph(
        ([0, 1, 2, 2, 3, 4], [1, 2, 0, 3, 4, 5]), idtype=idtype, device=device
    )
    y = F.tensor([0, 0, 0, 0, 1, 1])
    assert math.isclose(
        dgl.edge_label_informativeness(graph, y),
        0.25177597999572754,
        abs_tol=1e-6,
    )


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@parametrize_idtype
def test_node_label_informativeness(idtype):
    # IfChangeThenChange: python/dgl/label_informativeness.py
    # Update the docstring example.
    device = F.ctx()
    graph = dgl.graph(
        ([0, 1, 2, 2, 3, 4], [1, 2, 0, 3, 4, 5]), idtype=idtype, device=device
    )
    y = F.tensor([0, 0, 0, 0, 1, 1])
    assert math.isclose(
        dgl.node_label_informativeness(graph, y),
        0.3381872773170471,
        abs_tol=1e-6,
    )
