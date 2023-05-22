import unittest

import backend as F

import dgl
from dgl.data.movielens import MovieLensDataset


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="only supports pytorch"
)
def test_movielens():
    transform = dgl.AddSelfLoop(new_etypes=True)

    movielens = MovieLensDataset(
        name="ml-100k", valid_ratio=0.2, verbose=True
    )
    train_graph, valid_graph, test_graph = movielens[0]
    assert (
        train_graph.num_edges()
        + valid_graph.num_edges()
        + test_graph.num_edges()
        == 100000 * 2
    )
    assert train_graph.nodes['user'].data['feat'].shape[1] == \
        valid_graph.nodes['user'].data['feat'].shape[1] == \
        test_graph.nodes['user'].data['feat'].shape[1] == 23
    assert train_graph.nodes['movie'].data['feat'].shape[1] == \
        valid_graph.nodes['movie'].data['feat'].shape[1] == \
        test_graph.nodes['movie'].data['feat'].shape[1] == 320


    movielens1 = MovieLensDataset(
        name="ml-100k",
        valid_ratio=0.2,
        test_ratio=0.1,
        transform=transform,
        verbose=True
    )
    train_graph1, valid_graph1, test_graph1 = movielens1[0]
    assert (
        train_graph1.num_edges() - train_graph.num_edges()
        == train_graph.num_nodes()
    )
    assert (
        valid_graph1.num_edges() - valid_graph.num_edges()
        == valid_graph.num_nodes()
    )
    assert (
        test_graph1.num_edges() - test_graph.num_edges()
        == test_graph.num_nodes()
    )

    movielens = MovieLensDataset(
        name="ml-1m", valid_ratio=0.2, test_ratio=0.1, verbose=True
    )
    train_graph, valid_graph, test_graph = movielens[0]
    assert (
        train_graph.num_edges()
        + valid_graph.num_edges()
        + test_graph.num_edges()
        == 1000209 * 2
    )

    movielens = MovieLensDataset(
        name="ml-10m", valid_ratio=0.2, test_ratio=0.1, verbose=True
    )
    train_graph, valid_graph, test_graph = movielens[0]
    assert (
        train_graph.num_edges()
        + valid_graph.num_edges()
        + test_graph.num_edges()
        == 10000054 * 2
    )

