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
def test_chameleon():
    transform = dgl.AddSelfLoop(allow_duplicate=True)

    g = dgl.data.ChameleonDataset(force_reload=True)[0]
    assert g.num_nodes() == 2277
    assert g.num_edges() == 36101
    g2 = dgl.data.ChameleonDataset(force_reload=True, transform=transform)[0]
    assert g2.num_edges() - g.num_edges() == g.num_nodes()


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="only supports pytorch"
)
def test_squirrel():
    transform = dgl.AddSelfLoop(allow_duplicate=True)

    g = dgl.data.SquirrelDataset(force_reload=True)[0]
    assert g.num_nodes() == 5201
    assert g.num_edges() == 217073
    g2 = dgl.data.SquirrelDataset(force_reload=True, transform=transform)[0]
    assert g2.num_edges() - g.num_edges() == g.num_nodes()


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="only supports pytorch"
)
def test_cornell():
    transform = dgl.AddSelfLoop(allow_duplicate=True)

    g = dgl.data.CornellDataset(force_reload=True)[0]
    assert g.num_nodes() == 183
    assert g.num_edges() == 298
    g2 = dgl.data.CornellDataset(force_reload=True, transform=transform)[0]
    assert g2.num_edges() - g.num_edges() == g.num_nodes()


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="only supports pytorch"
)
def test_texas():
    transform = dgl.AddSelfLoop(allow_duplicate=True)

    g = dgl.data.TexasDataset(force_reload=True)[0]
    assert g.num_nodes() == 183
    assert g.num_edges() == 325
    g2 = dgl.data.TexasDataset(force_reload=True, transform=transform)[0]
    assert g2.num_edges() - g.num_edges() == g.num_nodes()


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="only supports pytorch"
)
def test_wisconsin():
    transform = dgl.AddSelfLoop(allow_duplicate=True)

    g = dgl.data.WisconsinDataset(force_reload=True)[0]
    assert g.num_nodes() == 251
    assert g.num_edges() == 515
    g2 = dgl.data.WisconsinDataset(force_reload=True, transform=transform)[0]
    assert g2.num_edges() - g.num_edges() == g.num_nodes()
