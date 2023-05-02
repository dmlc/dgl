import itertools
import math
import unittest
from collections import Counter

import backend as F

import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import pytest
import scipy.sparse as ssp
from dgl import DGLError
from dgl.ops import edge_softmax
from scipy.sparse import rand
from utils import get_cases, parametrize_idtype

edge_softmax_shapes = [(1,), (1, 3), (3, 4, 5)]
rfuncs = {"sum": fn.sum, "max": fn.max, "min": fn.min, "mean": fn.mean}
fill_value = {"sum": 0, "max": float("-inf")}
feat_size = 2


@pytest.mark.parametrize("g", get_cases(["clique"]))
@pytest.mark.parametrize("norm_by", ["src", "dst"])
@pytest.mark.parametrize("shp", edge_softmax_shapes)
@parametrize_idtype
def test_edge_softmax(g, norm_by, shp, idtype):
    g = g.astype(idtype).to(F.ctx())
    edata = F.tensor(np.random.rand(g.num_edges(), *shp))
    e1 = F.attach_grad(F.clone(edata))

    with F.record_grad():
        score1 = edge_softmax(g, e1, norm_by=norm_by)
        F.backward(F.reduce_sum(score1))
        grad_edata = F.grad(e1)

    with F.record_grad():
        e2 = F.attach_grad(F.clone(edata))
        e2_2d = F.reshape(
            e2,
            (g.number_of_src_nodes(), g.number_of_dst_nodes(), *e2.shape[1:]),
        )
        if norm_by == "src":
            score2 = F.softmax(e2_2d, 1)
            score2 = F.reshape(score2, (-1, *e2.shape[1:]))
        if norm_by == "dst":
            score2 = F.softmax(e2_2d, 0)
            score2 = F.reshape(score2, (-1, *e2.shape[1:]))
        assert F.allclose(score1, score2)
        print("forward passed")

        F.backward(F.reduce_sum(score2))
        assert F.allclose(F.grad(e2), grad_edata)
        print("backward passed")


def create_test_heterograph(idtype):
    # test heterograph from the docstring, plus a user -- wishes -- game relation
    # 3 users, 2 games, 2 developers
    # metagraph:
    #    ('user', 'follows', 'user'),
    #    ('user', 'plays', 'game'),
    #    ('user', 'wishes', 'game'),
    #    ('developer', 'develops', 'game')])

    g = dgl.heterograph(
        {
            ("user", "follows", "user"): ([0, 1, 2, 1, 1], [0, 0, 1, 1, 2]),
            ("user", "plays", "game"): ([0, 1, 2, 1], [0, 0, 1, 1]),
            ("user", "wishes", "game"): ([0, 1, 1], [0, 0, 1]),
            ("developer", "develops", "game"): ([0, 1, 0], [0, 1, 1]),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    assert g.idtype == idtype
    assert g.device == F.ctx()
    return g


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
def test_edge_softmax_unidirectional():
    g = dgl.heterograph(
        {
            ("A", "AB", "B"): (
                [1, 2, 3, 1, 2, 3, 1, 2, 3],
                [0, 0, 0, 1, 1, 1, 2, 2, 2],
            ),
            ("B", "BB", "B"): (
                [0, 1, 2, 0, 1, 2, 0, 1, 2],
                [0, 0, 0, 1, 1, 1, 2, 2, 2],
            ),
        }
    )
    g = g.to(F.ctx())
    g.edges["AB"].data["x"] = F.ones(9) * 2
    g.edges["BB"].data["x"] = F.ones(9)
    result = dgl.ops.edge_softmax(
        g, {"AB": g.edges["AB"].data["x"], "BB": g.edges["BB"].data["x"]}
    )

    ab = result["A", "AB", "B"]
    bb = result["B", "BB", "B"]
    e2 = F.zeros_like(ab) + math.exp(2) / ((math.exp(2) + math.exp(1)) * 3)
    e1 = F.zeros_like(bb) + math.exp(1) / ((math.exp(2) + math.exp(1)) * 3)
    assert F.allclose(ab, e2)
    assert F.allclose(bb, e1)


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@pytest.mark.parametrize("g", get_cases(["clique"]))
@pytest.mark.parametrize("norm_by", ["src", "dst"])
# @pytest.mark.parametrize('shp', edge_softmax_shapes)
@parametrize_idtype
def test_edge_softmax(g, norm_by, idtype):
    print("params", norm_by, idtype)

    g = create_test_heterograph(idtype)

    x1 = F.randn((g.num_edges("plays"), feat_size))
    x2 = F.randn((g.num_edges("follows"), feat_size))
    x3 = F.randn((g.num_edges("develops"), feat_size))
    x4 = F.randn((g.num_edges("wishes"), feat_size))

    F.attach_grad(F.clone(x1))
    F.attach_grad(F.clone(x2))
    F.attach_grad(F.clone(x3))
    F.attach_grad(F.clone(x4))

    g["plays"].edata["eid"] = x1
    g["follows"].edata["eid"] = x2
    g["develops"].edata["eid"] = x3
    g["wishes"].edata["eid"] = x4

    #################################################################
    #  edge_softmax() on homogeneous graph
    #################################################################

    with F.record_grad():
        hm_g = dgl.to_homogeneous(g)
        hm_x = F.cat((x3, x2, x1, x4), 0)
        hm_e = F.attach_grad(F.clone(hm_x))
        score_hm = edge_softmax(hm_g, hm_e, norm_by=norm_by)
        hm_g.edata["score"] = score_hm
        ht_g = dgl.to_heterogeneous(hm_g, g.ntypes, g.etypes)
        r1 = ht_g.edata["score"][("user", "plays", "game")]
        r2 = ht_g.edata["score"][("user", "follows", "user")]
        r3 = ht_g.edata["score"][("developer", "develops", "game")]
        r4 = ht_g.edata["score"][("user", "wishes", "game")]
        F.backward(F.reduce_sum(r1) + F.reduce_sum(r2))
        grad_edata_hm = F.grad(hm_e)

    #################################################################
    #  edge_softmax() on heterogeneous graph
    #################################################################

    e1 = F.attach_grad(F.clone(x1))
    e2 = F.attach_grad(F.clone(x2))
    e3 = F.attach_grad(F.clone(x3))
    e4 = F.attach_grad(F.clone(x4))
    e = {
        ("user", "follows", "user"): e2,
        ("user", "plays", "game"): e1,
        ("user", "wishes", "game"): e4,
        ("developer", "develops", "game"): e3,
    }
    with F.record_grad():
        score = edge_softmax(g, e, norm_by=norm_by)
        r5 = score[("user", "plays", "game")]
        r6 = score[("user", "follows", "user")]
        r7 = score[("developer", "develops", "game")]
        r8 = score[("user", "wishes", "game")]
        F.backward(F.reduce_sum(r5) + F.reduce_sum(r6))
        grad_edata_ht = F.cat(
            (F.grad(e3), F.grad(e2), F.grad(e1), F.grad(e4)), 0
        )
        # correctness check
        assert F.allclose(r1, r5)
        assert F.allclose(r2, r6)
        assert F.allclose(r3, r7)
        assert F.allclose(r4, r8)
        assert F.allclose(grad_edata_hm, grad_edata_ht)


if __name__ == "__main__":
    test_edge_softmax_unidirectional()
