import unittest

import backend as F

import dgl


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch",
    reason="Only supports PyTorch backend.",
)
def test_roman_empire():
    transform = dgl.AddSelfLoop(allow_duplicate=True)

    g = dgl.data.RomanEmpireDataset(force_reload=True)[0]
    assert g.num_nodes() == 22662
    assert g.num_edges() == 65854
    g2 = dgl.data.RomanEmpireDataset(force_reload=True, transform=transform)[0]
    assert g2.num_edges() - g.num_edges() == g.num_nodes()


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch",
    reason="Only supports PyTorch backend.",
)
def test_amazon_ratings():
    transform = dgl.AddSelfLoop(allow_duplicate=True)

    g = dgl.data.AmazonRatingsDataset(force_reload=True)[0]
    assert g.num_nodes() == 24492
    assert g.num_edges() == 186100
    g2 = dgl.data.AmazonRatingsDataset(force_reload=True, transform=transform)[
        0
    ]
    assert g2.num_edges() - g.num_edges() == g.num_nodes()


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch",
    reason="Only supports PyTorch backend.",
)
def test_minesweeper():
    transform = dgl.AddSelfLoop(allow_duplicate=True)

    g = dgl.data.MinesweeperDataset(force_reload=True)[0]
    assert g.num_nodes() == 10000
    assert g.num_edges() == 78804
    g2 = dgl.data.MinesweeperDataset(force_reload=True, transform=transform)[0]
    assert g2.num_edges() - g.num_edges() == g.num_nodes()


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch",
    reason="Only supports PyTorch backend.",
)
def test_tolokers():
    transform = dgl.AddSelfLoop(allow_duplicate=True)

    g = dgl.data.TolokersDataset(force_reload=True)[0]
    assert g.num_nodes() == 11758
    assert g.num_edges() == 1038000
    g2 = dgl.data.TolokersDataset(force_reload=True, transform=transform)[0]
    assert g2.num_edges() - g.num_edges() == g.num_nodes()


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch",
    reason="Only supports PyTorch backend.",
)
def test_questions():
    transform = dgl.AddSelfLoop(allow_duplicate=True)

    g = dgl.data.QuestionsDataset(force_reload=True)[0]
    assert g.num_nodes() == 48921
    assert g.num_edges() == 307080
    g2 = dgl.data.QuestionsDataset(force_reload=True, transform=transform)[0]
    assert g2.num_edges() - g.num_edges() == g.num_nodes()
