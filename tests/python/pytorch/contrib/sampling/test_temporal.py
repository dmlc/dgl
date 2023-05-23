import backend as F

import dgl
import numpy as np
import pytest
import unittest, torch
from utils import parametrize_idtype

dgl.seed(42)
np.random.seed(42)

@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="GPU sample neighbors not implemented",
)
@parametrize_idtype
@pytest.mark.parametrize('fanout', [2, 3, 5])
@pytest.mark.parametrize('replace', [False, True])
def test_homogeneous(idtype, fanout, replace):
    # node 0: some neighbors have smaller ts
    u0 = [101, 102, 103, 104, 105]
    v0 = [  0,   0,   0,   0,   0]
    t_u0 = [48, 49, 50, 51, 52]
    t_v0 = 50

    # node 1: all neighbors have larger ts (cannot be sampled)
    u1 = [106, 107, 108]
    v1 = [  1,   1,   1]
    t_u1 = [31, 32, 33]
    t_v1 = 30

    # node 2: all neighbors are valid
    u2 = [109, 110, 111]
    v2 = [  2,   2,   2]
    t_u2 = [40, 41, 42]
    t_v2 = 45

    ####
    N = max(max(u0), max(u1), max(u2)) + 1
    g = dgl.graph(
        (u0 + u1 + u2, v0 + v1 + v2),
        idtype=idtype,
        num_nodes=N
    )
    ts = torch.zeros(N).long()
    ts[0] = t_v0
    ts[1] = t_v1
    ts[2] = t_v2
    ts[u0] = torch.tensor(t_u0)
    ts[u1] = torch.tensor(t_u1)
    ts[u2] = torch.tensor(t_u2)

    ####
    nodes = torch.tensor([0, 1, 2])
    from dgl.contrib.sampling import temporal_sample_neighbors
    subg = temporal_sample_neighbors(
        g, nodes, fanout, ts, replace=replace)

    u, _ = subg.in_edges(0)
    assert all(ts[u.long()] <= t_v0)
    if replace:
        assert len(u) == fanout
    else:
        assert len(u) == min(3, fanout)

    u, _ = subg.in_edges(1)
    assert all(ts[u.long()] <= t_v1)
    assert len(u) == 0

    u, _ = subg.in_edges(2)
    assert all(ts[u.long()] <= t_v2)
    if replace:
        assert len(u) == fanout
    else:
        assert len(u) == min(3, fanout)
