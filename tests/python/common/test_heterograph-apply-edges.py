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
import scipy.sparse as spsp
import torch

from dgl import DGLError
from scipy.sparse import rand
from utils import get_cases, parametrize_idtype

rfuncs = {"sum": fn.sum, "max": fn.max, "min": fn.min, "mean": fn.mean}
fill_value = {"sum": 0, "max": float("-inf")}
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


def create_random_hetero_with_single_source_node_type(idtype):
    num_nodes = {"n1": 5, "n2": 10, "n3": 15}
    etypes = [("n1", "r1", "n2"), ("n1", "r2", "n3"), ("n1", "r3", "n2")]
    edges = {}
    for etype in etypes:
        src_ntype, _, dst_ntype = etype
        arr = spsp.random(
            num_nodes[src_ntype],
            num_nodes[dst_ntype],
            density=1,
            format="coo",
            random_state=100,
        )
        edges[etype] = (arr.row, arr.col)
    return dgl.heterograph(edges, idtype=idtype, device=F.ctx())


@parametrize_idtype
def test_unary_copy_u(idtype):
    def _test(mfunc):
        g = create_test_heterograph(idtype)

        x1 = F.randn((g.num_nodes("user"), feat_size))
        x2 = F.randn((g.num_nodes("developer"), feat_size))

        F.attach_grad(x1)
        F.attach_grad(x2)
        g.nodes["user"].data["h"] = x1
        g.nodes["developer"].data["h"] = x2

        #################################################################
        #  apply_edges() is called on each relation type separately
        #################################################################

        with F.record_grad():
            [
                g.apply_edges(fn.copy_u("h", "m"), etype=rel)
                for rel in g.canonical_etypes
            ]
            r1 = g["plays"].edata["m"]
            F.backward(r1, F.ones(r1.shape))
            n_grad1 = F.grad(g.ndata["h"]["user"])
        # TODO (Israt): clear not working
        g.edata["m"].clear()

        #################################################################
        #  apply_edges() is called on all relation types
        #################################################################

        g.apply_edges(fn.copy_u("h", "m"))
        r2 = g["plays"].edata["m"]
        F.backward(r2, F.ones(r2.shape))
        n_grad2 = F.grad(g.nodes["user"].data["h"])

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
        if not F.allclose(n_grad1, n_grad2):
            print("node grad")
            _print_error(n_grad1, n_grad2)
        assert F.allclose(n_grad1, n_grad2)

    _test(fn.copy_u)


@parametrize_idtype
def test_unary_copy_e(idtype):
    def _test(mfunc):
        g = create_test_heterograph(idtype)
        feat_size = 2

        x1 = F.randn((4, feat_size))
        x2 = F.randn((4, feat_size))
        x3 = F.randn((3, feat_size))
        x4 = F.randn((3, feat_size))
        F.attach_grad(x1)
        F.attach_grad(x2)
        F.attach_grad(x3)
        F.attach_grad(x4)
        g["plays"].edata["eid"] = x1
        g["follows"].edata["eid"] = x2
        g["develops"].edata["eid"] = x3
        g["wishes"].edata["eid"] = x4

        #################################################################
        #  apply_edges() is called on each relation type separately
        #################################################################
        with F.record_grad():
            [
                g.apply_edges(fn.copy_e("eid", "m"), etype=rel)
                for rel in g.canonical_etypes
            ]
            r1 = g["develops"].edata["m"]
            F.backward(r1, F.ones(r1.shape))
            e_grad1 = F.grad(g["develops"].edata["eid"])

        #################################################################
        #  apply_edges() is called on all relation types
        #################################################################

        g.apply_edges(fn.copy_e("eid", "m"))
        r2 = g["develops"].edata["m"]
        F.backward(r2, F.ones(r2.shape))
        e_grad2 = F.grad(g["develops"].edata["eid"])

        # # correctness check
        def _print_error(a, b):
            for i, (x, y) in enumerate(
                zip(F.asnumpy(a).flatten(), F.asnumpy(b).flatten())
            ):
                if not np.allclose(x, y):
                    print("@{} {} v.s. {}".format(i, x, y))

        if not F.allclose(r1, r2):
            _print_error(r1, r2)
        assert F.allclose(r1, r2)
        if not F.allclose(e_grad1, e_grad2):
            print("edge grad")
            _print_error(e_grad1, e_grad2)
        assert F.allclose(e_grad1, e_grad2)

    _test(fn.copy_e)


@parametrize_idtype
def test_binary_op(idtype):
    def _test(lhs, rhs, binary_op):
        g = create_test_heterograph(idtype)

        n1 = F.randn((g.num_nodes("user"), feat_size))
        n2 = F.randn((g.num_nodes("developer"), feat_size))
        n3 = F.randn((g.num_nodes("game"), feat_size))

        x1 = F.randn((g.num_edges("plays"), feat_size))
        x2 = F.randn((g.num_edges("follows"), feat_size))
        x3 = F.randn((g.num_edges("develops"), feat_size))
        x4 = F.randn((g.num_edges("wishes"), feat_size))

        builtin_msg_name = "{}_{}_{}".format(lhs, binary_op, rhs)
        builtin_msg = getattr(fn, builtin_msg_name)

        #################################################################
        #  apply_edges() is called on each relation type separately
        #################################################################

        F.attach_grad(n1)
        F.attach_grad(n2)
        F.attach_grad(n3)
        g.nodes["user"].data["h"] = n1
        g.nodes["developer"].data["h"] = n2
        g.nodes["game"].data["h"] = n3
        F.attach_grad(x1)
        F.attach_grad(x2)
        F.attach_grad(x3)
        F.attach_grad(x4)
        g["plays"].edata["h"] = x1
        g["follows"].edata["h"] = x2
        g["develops"].edata["h"] = x3
        g["wishes"].edata["h"] = x4

        with F.record_grad():
            [
                g.apply_edges(builtin_msg("h", "h", "m"), etype=rel)
                for rel in g.canonical_etypes
            ]
            r1 = g["plays"].edata["m"]
            loss = F.sum(r1.view(-1), 0)
            F.backward(loss)
            n_grad1 = F.grad(g.nodes["game"].data["h"])

        #################################################################
        #  apply_edges() is called on all relation types
        #################################################################

        F.attach_grad(n1)
        F.attach_grad(n2)
        F.attach_grad(n3)
        g.nodes["user"].data["h"] = n1
        g.nodes["developer"].data["h"] = n2
        g.nodes["game"].data["h"] = n3
        F.attach_grad(x1)
        F.attach_grad(x2)
        F.attach_grad(x3)
        F.attach_grad(x4)
        g["plays"].edata["h"] = x1
        g["follows"].edata["h"] = x2
        g["develops"].edata["h"] = x3
        g["wishes"].edata["h"] = x4

        with F.record_grad():
            g.apply_edges(builtin_msg("h", "h", "m"))
            r2 = g["plays"].edata["m"]
            loss = F.sum(r2.view(-1), 0)
            F.backward(loss)
            n_grad2 = F.grad(g.nodes["game"].data["h"])

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
        if n_grad1 is not None or n_grad2 is not None:
            if not F.allclose(n_grad1, n_grad2):
                print("node grad")
                _print_error(n_grad1, n_grad2)
            assert F.allclose(n_grad1, n_grad2)

    target = ["u", "v", "e"]
    for lhs, rhs in product(target, target):
        if lhs == rhs:
            continue
        for binary_op in ["add", "sub", "mul", "div", "dot"]:
            print(lhs, rhs, binary_op)
            _test(lhs, rhs, binary_op)


# Here we test heterograph with only single source node type because the format
# of node feature is a tensor.
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@parametrize_idtype
def test_heterograph_with_single_source_node_type_apply_edges(idtype):
    hg = create_random_hetero_with_single_source_node_type(idtype)

    hg.nodes["n1"].data["h"] = F.randn((hg.num_nodes("n1"), 1))
    hg.nodes["n2"].data["h"] = F.randn((hg.num_nodes("n2"), 1))
    hg.nodes["n3"].data["h"] = F.randn((hg.num_nodes("n3"), 1))

    assert type(hg.srcdata["h"]) == torch.Tensor
    hg.apply_edges(fn.u_add_v("h", "h", "x"))


if __name__ == "__main__":
    test_unary_copy_u()
    test_unary_copy_e()
