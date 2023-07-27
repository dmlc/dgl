import itertools
import unittest
from collections import Counter
from itertools import product

import backend as F

import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import pytest
import scipy.sparse as ssp
from dgl import DGLError
from scipy.sparse import rand
from utils import get_cases, parametrize_idtype

rfuncs = {"sum": fn.sum, "max": fn.max, "min": fn.min, "mean": fn.mean}
feat_size = 2


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
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
            ("user", "follows", "user"): ([0, 1, 2, 1], [0, 0, 1, 1]),
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


def create_test_heterograph_2(idtype):
    src = np.random.randint(0, 50, 25)
    dst = np.random.randint(0, 50, 25)
    src1 = np.random.randint(0, 25, 10)
    dst1 = np.random.randint(0, 25, 10)
    src2 = np.random.randint(0, 100, 1000)
    dst2 = np.random.randint(0, 100, 1000)
    g = dgl.heterograph(
        {
            ("user", "becomes", "player"): (src, dst),
            ("user", "follows", "user"): (src, dst),
            ("user", "plays", "game"): (src, dst),
            ("user", "wishes", "game"): (src1, dst1),
            ("developer", "develops", "game"): (src2, dst2),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    assert g.idtype == idtype
    assert g.device == F.ctx()
    return g


def create_test_heterograph_large(idtype):
    src = np.random.randint(0, 50, 2500)
    dst = np.random.randint(0, 50, 2500)
    g = dgl.heterograph(
        {
            ("user", "follows", "user"): (src, dst),
            ("user", "plays", "game"): (src, dst),
            ("user", "wishes", "game"): (src, dst),
            ("developer", "develops", "game"): (src, dst),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    assert g.idtype == idtype
    assert g.device == F.ctx()
    return g


@parametrize_idtype
def test_unary_copy_u(idtype):
    def _test(mfunc, rfunc):
        g = create_test_heterograph_2(idtype)
        g0 = create_test_heterograph(idtype)
        g1 = create_test_heterograph_large(idtype)
        cross_reducer = rfunc.__name__
        x1 = F.randn((g.num_nodes("user"), feat_size))
        x2 = F.randn((g.num_nodes("developer"), feat_size))
        F.attach_grad(x1)
        F.attach_grad(x2)
        g.nodes["user"].data["h"] = x1
        g.nodes["developer"].data["h"] = x2

        #################################################################
        #  multi_update_all(): call msg_passing separately for each etype
        #################################################################

        with F.record_grad():
            g.multi_update_all(
                {
                    etype: (mfunc("h", "m"), rfunc("m", "y"))
                    for etype in g.canonical_etypes
                },
                cross_reducer,
            )
            r1 = g.nodes["game"].data["y"].clone()
            r2 = g.nodes["user"].data["y"].clone()
            r3 = g.nodes["player"].data["y"].clone()
            loss = r1.sum() + r2.sum() + r3.sum()
            F.backward(loss)
            n_grad1 = F.grad(g.nodes["user"].data["h"]).clone()
            n_grad2 = F.grad(g.nodes["developer"].data["h"]).clone()

        g.nodes["user"].data.clear()
        g.nodes["developer"].data.clear()
        g.nodes["game"].data.clear()
        g.nodes["player"].data.clear()

        #################################################################
        #  update_all(): call msg_passing for all etypes
        #################################################################

        F.attach_grad(x1)
        F.attach_grad(x2)
        g.nodes["user"].data["h"] = x1
        g.nodes["developer"].data["h"] = x2

        with F.record_grad():
            g.update_all(mfunc("h", "m"), rfunc("m", "y"))
            r4 = g.nodes["game"].data["y"]
            r5 = g.nodes["user"].data["y"]
            r6 = g.nodes["player"].data["y"]
            loss = r4.sum() + r5.sum() + r6.sum()
            F.backward(loss)
            n_grad3 = F.grad(g.nodes["user"].data["h"])
            n_grad4 = F.grad(g.nodes["developer"].data["h"])

        assert F.allclose(r1, r4)
        assert F.allclose(r2, r5)
        assert F.allclose(r3, r6)
        assert F.allclose(n_grad1, n_grad3)
        assert F.allclose(n_grad2, n_grad4)

    _test(fn.copy_u, fn.sum)
    _test(fn.copy_u, fn.max)
    _test(fn.copy_u, fn.min)
    # _test('copy_u', 'mean')


@parametrize_idtype
def test_unary_copy_e(idtype):
    def _test(mfunc, rfunc):
        g = create_test_heterograph_large(idtype)
        g0 = create_test_heterograph_2(idtype)
        g1 = create_test_heterograph(idtype)
        cross_reducer = rfunc.__name__
        x1 = F.randn((g.num_edges("plays"), feat_size))
        x2 = F.randn((g.num_edges("follows"), feat_size))
        x3 = F.randn((g.num_edges("develops"), feat_size))
        x4 = F.randn((g.num_edges("wishes"), feat_size))
        F.attach_grad(x1)
        F.attach_grad(x2)
        F.attach_grad(x3)
        F.attach_grad(x4)
        g["plays"].edata["eid"] = x1
        g["follows"].edata["eid"] = x2
        g["develops"].edata["eid"] = x3
        g["wishes"].edata["eid"] = x4

        #################################################################
        #  multi_update_all(): call msg_passing separately for each etype
        #################################################################

        with F.record_grad():
            g.multi_update_all(
                {
                    "plays": (mfunc("eid", "m"), rfunc("m", "y")),
                    "follows": (mfunc("eid", "m"), rfunc("m", "y")),
                    "develops": (mfunc("eid", "m"), rfunc("m", "y")),
                    "wishes": (mfunc("eid", "m"), rfunc("m", "y")),
                },
                cross_reducer,
            )
            r1 = g.nodes["game"].data["y"].clone()
            r2 = g.nodes["user"].data["y"].clone()
            loss = r1.sum() + r2.sum()
            F.backward(loss)
            e_grad1 = F.grad(g["develops"].edata["eid"]).clone()
            e_grad2 = F.grad(g["plays"].edata["eid"]).clone()
            e_grad3 = F.grad(g["wishes"].edata["eid"]).clone()
            e_grad4 = F.grad(g["follows"].edata["eid"]).clone()
        {etype: (g[etype].edata.clear()) for _, etype, _ in g.canonical_etypes},

        #################################################################
        #  update_all(): call msg_passing for all etypes
        #################################################################

        # TODO(Israt): output type can be None in multi_update and empty
        F.attach_grad(x1)
        F.attach_grad(x2)
        F.attach_grad(x3)
        F.attach_grad(x4)

        g["plays"].edata["eid"] = x1
        g["follows"].edata["eid"] = x2
        g["develops"].edata["eid"] = x3
        g["wishes"].edata["eid"] = x4

        with F.record_grad():
            g.update_all(mfunc("eid", "m"), rfunc("m", "y"))
            r3 = g.nodes["game"].data["y"]
            r4 = g.nodes["user"].data["y"]
            loss = r3.sum() + r4.sum()
            F.backward(loss)
            e_grad5 = F.grad(g["develops"].edata["eid"])
            e_grad6 = F.grad(g["plays"].edata["eid"])
            e_grad7 = F.grad(g["wishes"].edata["eid"])
            e_grad8 = F.grad(g["follows"].edata["eid"])

        # # correctness check
        def _print_error(a, b):
            for i, (x, y) in enumerate(
                zip(F.asnumpy(a).flatten(), F.asnumpy(b).flatten())
            ):
                if not np.allclose(x, y):
                    print("@{} {} v.s. {}".format(i, x, y))

        assert F.allclose(r1, r3)
        assert F.allclose(r2, r4)
        assert F.allclose(e_grad1, e_grad5)
        assert F.allclose(e_grad2, e_grad6)
        assert F.allclose(e_grad3, e_grad7)
        assert F.allclose(e_grad4, e_grad8)

    _test(fn.copy_e, fn.sum)
    _test(fn.copy_e, fn.max)
    _test(fn.copy_e, fn.min)
    # _test('copy_e', 'mean')


@parametrize_idtype
def test_binary_op(idtype):
    def _test(lhs, rhs, binary_op, reducer):
        g = create_test_heterograph(idtype)

        x1 = F.randn((g.num_nodes("user"), feat_size))
        x2 = F.randn((g.num_nodes("developer"), feat_size))
        x3 = F.randn((g.num_nodes("game"), feat_size))

        F.attach_grad(x1)
        F.attach_grad(x2)
        F.attach_grad(x3)
        g.nodes["user"].data["h"] = x1
        g.nodes["developer"].data["h"] = x2
        g.nodes["game"].data["h"] = x3

        x1 = F.randn((4, feat_size))
        x2 = F.randn((4, feat_size))
        x3 = F.randn((3, feat_size))
        x4 = F.randn((3, feat_size))
        F.attach_grad(x1)
        F.attach_grad(x2)
        F.attach_grad(x3)
        F.attach_grad(x4)
        g["plays"].edata["h"] = x1
        g["follows"].edata["h"] = x2
        g["develops"].edata["h"] = x3
        g["wishes"].edata["h"] = x4

        builtin_msg_name = "{}_{}_{}".format(lhs, binary_op, rhs)
        builtin_msg = getattr(fn, builtin_msg_name)
        builtin_red = getattr(fn, reducer)

        #################################################################
        #  multi_update_all(): call msg_passing separately for each etype
        #################################################################

        with F.record_grad():
            g.multi_update_all(
                {
                    etype: (builtin_msg("h", "h", "m"), builtin_red("m", "y"))
                    for etype in g.canonical_etypes
                },
                "sum",
            )
            r1 = g.nodes["game"].data["y"]
            F.backward(r1, F.ones(r1.shape))
            n_grad1 = F.grad(r1)

        #################################################################
        #  update_all(): call msg_passing for all etypes
        #################################################################

        g.update_all(builtin_msg("h", "h", "m"), builtin_red("m", "y"))
        r2 = g.nodes["game"].data["y"]
        F.backward(r2, F.ones(r2.shape))
        n_grad2 = F.grad(r2)

        # correctness check
        def _print_error(a, b):
            for i, (x, y) in enumerate(
                zip(F.asnumpy(a).flatten(), F.asnumpy(b).flatten())
            ):
                if not np.allclose(x, y):
                    print("@{} {} v.s. {}".format(i, x, y))

        if not F.allclose(r1, r2):
            _print_error(r1, r2)
        assert F.allclose(r1, r2)
        # TODO (Israt): r1 and r2 have different frad func associated with
        # if not F.allclose(n_grad1, n_grad2):
        #     print('node grad')
        #     _print_error(n_grad1, n_grad2)
        # assert(F.allclose(n_grad1, n_grad2))

    target = ["u", "v", "e"]
    for lhs, rhs in product(target, target):
        if lhs == rhs:
            continue
        for binary_op in ["add", "sub", "mul", "div"]:
            # TODO(Israt) :Add support for reduce func "max", "min", "mean"
            for reducer in ["sum"]:
                print(lhs, rhs, binary_op, reducer)
                _test(lhs, rhs, binary_op, reducer)


# Issue #5873
def test_multi_update_all_minmax_reduce_with_isolated_nodes():
    g = dgl.heterograph(
        {
            ("A", "AB", "B"): ([0, 1, 2, 3], [0, 0, 1, 1]),
            ("C", "CB", "B"): ([0, 1, 2, 3], [2, 2, 3, 3]),
        },
        device=F.ctx(),
    )
    g.nodes["A"].data["x"] = F.randn((4, 16))
    g.nodes["C"].data["x"] = F.randn((4, 16))
    g.multi_update_all(
        {
            "AB": (dgl.function.copy_u("x", "m"), dgl.function.min("m", "a1")),
            "CB": (dgl.function.copy_u("x", "m"), dgl.function.min("m", "a2")),
        },
        cross_reducer="min",
    )
    assert not np.isinf(F.asnumpy(g.nodes["B"].data["a1"])).any()
    assert not np.isinf(F.asnumpy(g.nodes["B"].data["a2"])).any()

    g.multi_update_all(
        {
            "AB": (dgl.function.copy_u("x", "m"), dgl.function.max("m", "a1")),
            "CB": (dgl.function.copy_u("x", "m"), dgl.function.max("m", "a2")),
        },
        cross_reducer="max",
    )
    assert not np.isinf(F.asnumpy(g.nodes["B"].data["a1"])).any()
    assert not np.isinf(F.asnumpy(g.nodes["B"].data["a2"])).any()


if __name__ == "__main__":
    test_unary_copy_u()
    test_unary_copy_e()
    test_binary_op()
