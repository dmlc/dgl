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
def test_movielens():
    transform = dgl.AddSelfLoop(allow_duplicate=True)

    movielens = dgl.data.MovieLensDataset(
        name="ml-100k", valid_ratio=0.2, test_ratio=0.1, force_reload=True
    )
    train_graph, valid_graph, test_graph = movielens[0]
    assert (
        train_graph.num_nodes()
        + valid_graph.num_nodes()
        + test_graph.num_nodes()
        == 943 + 1682
    )
    assert (
        train_graph.num_edges()
        + valid_graph.num_edges()
        + test_graph.num_edges()
        == 100000 * 2
    )

    movielens1 = dgl.data.MovieLensDataset(
        name="ml-100k",
        valid_ratio=0.2,
        test_ratio=0.1,
        force_reload=True,
        transform=transform,
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

    train_ratings, valid_ratings, test_ratings = (
        movielens.info["train_rating_pairs"],
        movielens.info["valid_rating_pairs"],
        movielens.info["test_rating_pairs"],
    )
    assert (
        len(train_ratings[0])
        == len(train_ratings[1])
        == train_graph.num_edges() / 2
    )
    assert (
        len(valid_ratings[0])
        == len(valid_ratings[1])
        == valid_graph.num_edges() / 2
    )
    assert (
        len(test_ratings[0])
        == len(test_ratings[1])
        == test_graph.num_edges() / 2
    )

    user_feat, movie_feat = (
        movielens.feat["user_feat"],
        movielens.feat["movie_feat"],
    )
    assert user_feat.shape[0] + movie_feat.shape[0] == 943 + 1682
