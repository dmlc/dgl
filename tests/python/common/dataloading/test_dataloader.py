import unittest

import backend as F

import dgl
from dgl.dataloading import (
    as_edge_prediction_sampler,
    negative_sampler,
    NeighborSampler,
)
from utils import parametrize_idtype


def create_test_graph(idtype):
    # test heterograph from the docstring, plus a user -- wishes -- game relation
    # 3 users, 2 games, 2 developers
    # metagraph:
    #    ('user', 'follows', 'user'),
    #    ('user', 'plays', 'game'),
    #    ('user', 'wishes', 'game'),
    #    ('developer', 'develops', 'game')])

    g = dgl.heterograph(
        {
            ("user", "follows", "user"): ([0, 1], [1, 2]),
            ("user", "plays", "game"): ([0, 1, 2, 1], [0, 0, 1, 1]),
            ("user", "wishes", "game"): ([0, 2], [1, 0]),
            ("developer", "develops", "game"): ([0, 1], [0, 1]),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    assert g.idtype == idtype
    assert g.device == F.ctx()
    return g


@parametrize_idtype
def test_edge_prediction_sampler(idtype):
    g = create_test_graph(idtype)
    sampler = NeighborSampler([10, 10])
    sampler = as_edge_prediction_sampler(
        sampler, negative_sampler=negative_sampler.Uniform(1)
    )

    seeds = F.copy_to(F.arange(0, 2, dtype=idtype), ctx=F.ctx())
    # just a smoke test to make sure we don't fail internal assertions
    result = sampler.sample(g, {"follows": seeds})


if __name__ == "__main__":
    test_edge_prediction_sampler()
