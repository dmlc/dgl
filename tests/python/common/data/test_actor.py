import unittest

import backend as F

import dgl


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="only supports pytorch"
)
def test_actor():
    transform = dgl.AddSelfLoop(allow_duplicate=True)

    g = dgl.data.ActorDataset(force_reload=True)[0]
    assert g.num_nodes() == 7600
    assert g.num_edges() == 33391
    g2 = dgl.data.ActorDataset(force_reload=True, transform=transform)[0]
    assert g2.num_edges() - g.num_edges() == g.num_nodes()
