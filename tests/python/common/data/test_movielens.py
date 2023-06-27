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

    movielens = MovieLensDataset(name="ml-100k", valid_ratio=0.2, verbose=True)
    g = movielens[0]
    assert g.num_edges("user-movie") == g.num_edges("movie-user") == 100000
    assert (
        g.nodes["user"].data["feat"].shape[1]
        == g.nodes["user"].data["feat"].shape[1]
        == g.nodes["user"].data["feat"].shape[1]
        == 23
    )
    assert (
        g.nodes["movie"].data["feat"].shape[1]
        == g.nodes["movie"].data["feat"].shape[1]
        == g.nodes["movie"].data["feat"].shape[1]
        == 320
    )

    movielens = MovieLensDataset(
        name="ml-100k", valid_ratio=0.2, transform=transform, verbose=True
    )
    g1 = movielens[0]
    assert g1.num_edges() - g.num_edges() == g.num_nodes()
    assert g1.num_edges() - g.num_edges() == g.num_nodes()
    assert g1.num_edges() - g.num_edges() == g.num_nodes()

    movielens = MovieLensDataset(
        name="ml-1m", valid_ratio=0.2, test_ratio=0.1, verbose=True
    )
    g = movielens[0]
    assert g.num_edges("user-movie") == g.num_edges("movie-user") == 1000209

    movielens = MovieLensDataset(
        name="ml-10m", valid_ratio=0.2, test_ratio=0.1, verbose=True
    )
    g = movielens[0]
    assert g.num_edges("user-movie") == g.num_edges("movie-user") == 10000054
