import backend as F

import dgl
from test_utils import parametrize_idtype


@parametrize_idtype
def test_node_homophily(idtype):
    # NOTE: If you want to update this test case, remember to update the
    # docstring example too.
    device = F.ctx()
    g = dgl.graph(([1, 2, 0, 4], [0, 1, 2, 3]), idtype=idtype, device=device)
    y = F.tensor([0, 0, 0, 0, 1])
    assert dgl.node_homophily(graph, y) == 0.6000000238418579
