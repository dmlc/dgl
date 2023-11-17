##
#   Copyright 2019-2021 Contributors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import math
import os
import unittest

import backend as F

import dgl
import dgl.function as fn
import dgl.partition
import networkx as nx
import numpy as np
import pytest
from scipy import sparse as spsp
from utils import parametrize_idtype
from utils.graph_cases import get_cases

D = 5


def create_test_heterograph3(idtype):
    g = dgl.heterograph(
        {
            ("user", "plays", "game"): (
                F.tensor([0, 1, 1, 2], dtype=idtype),
                F.tensor([0, 0, 1, 1], dtype=idtype),
            ),
            ("developer", "develops", "game"): (
                F.tensor([0, 1], dtype=idtype),
                F.tensor([0, 1], dtype=idtype),
            ),
        },
        idtype=idtype,
        device=F.ctx(),
    )

    g.nodes["user"].data["h"] = F.copy_to(
        F.tensor([1, 1, 1], dtype=idtype), ctx=F.ctx()
    )
    g.nodes["game"].data["h"] = F.copy_to(
        F.tensor([2, 2], dtype=idtype), ctx=F.ctx()
    )
    g.nodes["developer"].data["h"] = F.copy_to(
        F.tensor([3, 3], dtype=idtype), ctx=F.ctx()
    )
    g.edges["plays"].data["h"] = F.copy_to(
        F.tensor([1, 1, 1, 1], dtype=idtype), ctx=F.ctx()
    )
    return g


def create_test_heterograph4(idtype):
    g = dgl.heterograph(
        {
            ("user", "follows", "user"): (
                F.tensor([0, 1, 1, 2, 2, 2], dtype=idtype),
                F.tensor([0, 0, 1, 1, 2, 2], dtype=idtype),
            ),
            ("user", "plays", "game"): (
                F.tensor([0, 1], dtype=idtype),
                F.tensor([0, 1], dtype=idtype),
            ),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    g.nodes["user"].data["h"] = F.copy_to(
        F.tensor([1, 1, 1], dtype=idtype), ctx=F.ctx()
    )
    g.nodes["game"].data["h"] = F.copy_to(
        F.tensor([2, 2], dtype=idtype), ctx=F.ctx()
    )
    g.edges["follows"].data["h"] = F.copy_to(
        F.tensor([1, 2, 3, 4, 5, 6], dtype=idtype), ctx=F.ctx()
    )
    g.edges["plays"].data["h"] = F.copy_to(
        F.tensor([1, 2], dtype=idtype), ctx=F.ctx()
    )
    return g


def create_test_heterograph5(idtype):
    g = dgl.heterograph(
        {
            ("user", "follows", "user"): (
                F.tensor([1, 2], dtype=idtype),
                F.tensor([0, 1], dtype=idtype),
            ),
            ("user", "plays", "game"): (
                F.tensor([0, 1], dtype=idtype),
                F.tensor([0, 1], dtype=idtype),
            ),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    g.nodes["user"].data["h"] = F.copy_to(
        F.tensor([1, 1, 1], dtype=idtype), ctx=F.ctx()
    )
    g.nodes["game"].data["h"] = F.copy_to(
        F.tensor([2, 2], dtype=idtype), ctx=F.ctx()
    )
    g.edges["follows"].data["h"] = F.copy_to(
        F.tensor([1, 2], dtype=idtype), ctx=F.ctx()
    )
    g.edges["plays"].data["h"] = F.copy_to(
        F.tensor([1, 2], dtype=idtype), ctx=F.ctx()
    )
    return g


# line graph related


def test_line_graph1():
    N = 5
    G = dgl.from_networkx(nx.star_graph(N)).to(F.ctx())
    G.edata["h"] = F.randn((2 * N, D))
    L = G.line_graph(shared=True)
    assert L.num_nodes() == 2 * N
    assert F.allclose(L.ndata["h"], G.edata["h"])
    assert G.device == F.ctx()


@parametrize_idtype
def test_line_graph2(idtype):
    g = dgl.heterograph(
        {("user", "follows", "user"): ([0, 1, 1, 2, 2], [2, 0, 2, 0, 1])},
        idtype=idtype,
    )
    lg = dgl.line_graph(g)
    assert lg.num_nodes() == 5
    assert lg.num_edges() == 8
    row, col = lg.edges()
    assert np.array_equal(F.asnumpy(row), np.array([0, 0, 1, 2, 2, 3, 4, 4]))
    assert np.array_equal(F.asnumpy(col), np.array([3, 4, 0, 3, 4, 0, 1, 2]))

    lg = dgl.line_graph(g, backtracking=False)
    assert lg.num_nodes() == 5
    assert lg.num_edges() == 4
    row, col = lg.edges()
    assert np.array_equal(F.asnumpy(row), np.array([0, 1, 2, 4]))
    assert np.array_equal(F.asnumpy(col), np.array([4, 0, 3, 1]))
    g = dgl.heterograph(
        {("user", "follows", "user"): ([0, 1, 1, 2, 2], [2, 0, 2, 0, 1])},
        idtype=idtype,
    ).formats("csr")
    lg = dgl.line_graph(g)
    assert lg.num_nodes() == 5
    assert lg.num_edges() == 8
    row, col = lg.edges()
    assert np.array_equal(F.asnumpy(row), np.array([0, 0, 1, 2, 2, 3, 4, 4]))
    assert np.array_equal(F.asnumpy(col), np.array([3, 4, 0, 3, 4, 0, 1, 2]))

    g = dgl.heterograph(
        {("user", "follows", "user"): ([0, 1, 1, 2, 2], [2, 0, 2, 0, 1])},
        idtype=idtype,
    ).formats("csc")
    lg = dgl.line_graph(g)
    assert lg.num_nodes() == 5
    assert lg.num_edges() == 8
    row, col, eid = lg.edges("all")
    row = F.asnumpy(row)
    col = F.asnumpy(col)
    eid = F.asnumpy(eid).astype(int)
    order = np.argsort(eid)
    assert np.array_equal(row[order], np.array([0, 0, 1, 2, 2, 3, 4, 4]))
    assert np.array_equal(col[order], np.array([3, 4, 0, 3, 4, 0, 1, 2]))


def test_no_backtracking():
    N = 5
    G = dgl.from_networkx(nx.star_graph(N))
    L = G.line_graph(backtracking=False)
    assert L.num_nodes() == 2 * N
    for i in range(1, N):
        e1 = G.edge_ids(0, i)
        e2 = G.edge_ids(i, 0)
        assert not L.has_edges_between(e1, e2)
        assert not L.has_edges_between(e2, e1)


# reverse graph related
@parametrize_idtype
def test_reverse(idtype):
    g = dgl.graph([])
    g = g.astype(idtype).to(F.ctx())
    g.add_nodes(5)
    # The graph need not to be completely connected.
    g.add_edges([0, 1, 2], [1, 2, 1])
    g.ndata["h"] = F.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
    g.edata["h"] = F.tensor([[5.0], [6.0], [7.0]])
    rg = g.reverse()

    assert g.is_multigraph == rg.is_multigraph

    assert g.num_nodes() == rg.num_nodes()
    assert g.num_edges() == rg.num_edges()
    assert F.allclose(
        F.astype(rg.has_edges_between([1, 2, 1], [0, 1, 2]), F.float32),
        F.ones((3,)),
    )
    assert g.edge_ids(0, 1) == rg.edge_ids(1, 0)
    assert g.edge_ids(1, 2) == rg.edge_ids(2, 1)
    assert g.edge_ids(2, 1) == rg.edge_ids(1, 2)

    # test dgl.reverse
    # test homogeneous graph
    g = dgl.graph((F.tensor([0, 1, 2]), F.tensor([1, 2, 0])))
    g.ndata["h"] = F.tensor([[0.0], [1.0], [2.0]])
    g.edata["h"] = F.tensor([[3.0], [4.0], [5.0]])
    g_r = dgl.reverse(g)
    assert g.num_nodes() == g_r.num_nodes()
    assert g.num_edges() == g_r.num_edges()
    u_g, v_g, eids_g = g.all_edges(form="all")
    u_rg, v_rg, eids_rg = g_r.all_edges(form="all")
    assert F.array_equal(u_g, v_rg)
    assert F.array_equal(v_g, u_rg)
    assert F.array_equal(eids_g, eids_rg)
    assert F.array_equal(g.ndata["h"], g_r.ndata["h"])
    assert len(g_r.edata) == 0

    # without share ndata
    g_r = dgl.reverse(g, copy_ndata=False)
    assert g.num_nodes() == g_r.num_nodes()
    assert g.num_edges() == g_r.num_edges()
    assert len(g_r.ndata) == 0
    assert len(g_r.edata) == 0

    # with share ndata and edata
    g_r = dgl.reverse(g, copy_ndata=True, copy_edata=True)
    assert g.num_nodes() == g_r.num_nodes()
    assert g.num_edges() == g_r.num_edges()
    assert F.array_equal(g.ndata["h"], g_r.ndata["h"])
    assert F.array_equal(g.edata["h"], g_r.edata["h"])

    # add new node feature to g_r
    g_r.ndata["hh"] = F.tensor([0, 1, 2])
    assert ("hh" in g.ndata) is False
    assert ("hh" in g_r.ndata) is True

    # add new edge feature to g_r
    g_r.edata["hh"] = F.tensor([0, 1, 2])
    assert ("hh" in g.edata) is False
    assert ("hh" in g_r.edata) is True

    # test heterogeneous graph
    g = dgl.heterograph(
        {
            ("user", "follows", "user"): (
                [0, 1, 2, 4, 3, 1, 3],
                [1, 2, 3, 2, 0, 0, 1],
            ),
            ("user", "plays", "game"): (
                [0, 0, 2, 3, 3, 4, 1],
                [1, 0, 1, 0, 1, 0, 0],
            ),
            ("developer", "develops", "game"): ([0, 1, 1, 2], [0, 0, 1, 1]),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    g.nodes["user"].data["h"] = F.tensor([0, 1, 2, 3, 4])
    g.nodes["user"].data["hh"] = F.tensor([1, 1, 1, 1, 1])
    g.nodes["game"].data["h"] = F.tensor([0, 1])
    g.edges["follows"].data["h"] = F.tensor([0, 1, 2, 4, 3, 1, 3])
    g.edges["follows"].data["hh"] = F.tensor([1, 2, 3, 2, 0, 0, 1])
    g_r = dgl.reverse(g)

    for etype_g, etype_gr in zip(g.canonical_etypes, g_r.canonical_etypes):
        assert etype_g[0] == etype_gr[2]
        assert etype_g[1] == etype_gr[1]
        assert etype_g[2] == etype_gr[0]
        assert g.num_edges(etype_g) == g_r.num_edges(etype_gr)
    for ntype in g.ntypes:
        assert g.num_nodes(ntype) == g_r.num_nodes(ntype)
    assert F.array_equal(g.nodes["user"].data["h"], g_r.nodes["user"].data["h"])
    assert F.array_equal(
        g.nodes["user"].data["hh"], g_r.nodes["user"].data["hh"]
    )
    assert F.array_equal(g.nodes["game"].data["h"], g_r.nodes["game"].data["h"])
    assert len(g_r.edges["follows"].data) == 0
    u_g, v_g, eids_g = g.all_edges(
        form="all", etype=("user", "follows", "user")
    )
    u_rg, v_rg, eids_rg = g_r.all_edges(
        form="all", etype=("user", "follows", "user")
    )
    assert F.array_equal(u_g, v_rg)
    assert F.array_equal(v_g, u_rg)
    assert F.array_equal(eids_g, eids_rg)
    u_g, v_g, eids_g = g.all_edges(form="all", etype=("user", "plays", "game"))
    u_rg, v_rg, eids_rg = g_r.all_edges(
        form="all", etype=("game", "plays", "user")
    )
    assert F.array_equal(u_g, v_rg)
    assert F.array_equal(v_g, u_rg)
    assert F.array_equal(eids_g, eids_rg)
    u_g, v_g, eids_g = g.all_edges(
        form="all", etype=("developer", "develops", "game")
    )
    u_rg, v_rg, eids_rg = g_r.all_edges(
        form="all", etype=("game", "develops", "developer")
    )
    assert F.array_equal(u_g, v_rg)
    assert F.array_equal(v_g, u_rg)
    assert F.array_equal(eids_g, eids_rg)

    # withour share ndata
    g_r = dgl.reverse(g, copy_ndata=False)
    for etype_g, etype_gr in zip(g.canonical_etypes, g_r.canonical_etypes):
        assert etype_g[0] == etype_gr[2]
        assert etype_g[1] == etype_gr[1]
        assert etype_g[2] == etype_gr[0]
        assert g.num_edges(etype_g) == g_r.num_edges(etype_gr)
    for ntype in g.ntypes:
        assert g.num_nodes(ntype) == g_r.num_nodes(ntype)
    assert len(g_r.nodes["user"].data) == 0
    assert len(g_r.nodes["game"].data) == 0

    g_r = dgl.reverse(g, copy_ndata=True, copy_edata=True)
    print(g_r)
    for etype_g, etype_gr in zip(g.canonical_etypes, g_r.canonical_etypes):
        assert etype_g[0] == etype_gr[2]
        assert etype_g[1] == etype_gr[1]
        assert etype_g[2] == etype_gr[0]
        assert g.num_edges(etype_g) == g_r.num_edges(etype_gr)
    assert F.array_equal(
        g.edges["follows"].data["h"], g_r.edges["follows"].data["h"]
    )
    assert F.array_equal(
        g.edges["follows"].data["hh"], g_r.edges["follows"].data["hh"]
    )

    # add new node feature to g_r
    g_r.nodes["user"].data["hhh"] = F.tensor([0, 1, 2, 3, 4])
    assert ("hhh" in g.nodes["user"].data) is False
    assert ("hhh" in g_r.nodes["user"].data) is True

    # add new edge feature to g_r
    g_r.edges["follows"].data["hhh"] = F.tensor([1, 2, 3, 2, 0, 0, 1])
    assert ("hhh" in g.edges["follows"].data) is False
    assert ("hhh" in g_r.edges["follows"].data) is True


@parametrize_idtype
def test_reverse_shared_frames(idtype):
    g = dgl.graph([])
    g = g.astype(idtype).to(F.ctx())
    g.add_nodes(3)
    g.add_edges([0, 1, 2], [1, 2, 1])
    g.ndata["h"] = F.tensor([[0.0], [1.0], [2.0]])
    g.edata["h"] = F.tensor([[3.0], [4.0], [5.0]])

    rg = g.reverse(copy_ndata=True, copy_edata=True)
    assert F.allclose(g.ndata["h"], rg.ndata["h"])
    assert F.allclose(g.edata["h"], rg.edata["h"])
    assert F.allclose(
        g.edges[[0, 2], [1, 1]].data["h"], rg.edges[[1, 1], [0, 2]].data["h"]
    )


@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
def test_to_bidirected():
    # homogeneous graph
    elist = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2)]
    num_edges = 7
    g = dgl.graph(tuple(zip(*elist)))
    elist.append((1, 2))
    elist = set(elist)
    big = dgl.to_bidirected(g)
    assert big.num_edges() == num_edges
    src, dst = big.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == set(elist)

    # heterogeneous graph
    elist1 = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2)]
    elist2 = [(0, 0), (0, 1)]
    g = dgl.heterograph(
        {
            ("user", "wins", "user"): tuple(zip(*elist1)),
            ("user", "follows", "user"): tuple(zip(*elist2)),
        }
    )
    g.nodes["user"].data["h"] = F.ones((3, 1))
    elist1.append((1, 2))
    elist1 = set(elist1)
    elist2.append((1, 0))
    elist2 = set(elist2)
    big = dgl.to_bidirected(g)
    assert big.num_edges("wins") == 7
    assert big.num_edges("follows") == 3
    src, dst = big.edges(etype="wins")
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == set(elist1)
    src, dst = big.edges(etype="follows")
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == set(elist2)

    big = dgl.to_bidirected(g, copy_ndata=True)
    assert F.array_equal(g.nodes["user"].data["h"], big.nodes["user"].data["h"])


def test_add_reverse_edges():
    # homogeneous graph
    g = dgl.graph((F.tensor([0, 1, 3, 1]), F.tensor([1, 2, 0, 2])))
    g.ndata["h"] = F.tensor([[0.0], [1.0], [2.0], [1.0]])
    g.edata["h"] = F.tensor([[3.0], [4.0], [5.0], [6.0]])
    bg = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
    u, v = g.edges()
    ub, vb = bg.edges()
    assert F.array_equal(F.cat([u, v], dim=0), ub)
    assert F.array_equal(F.cat([v, u], dim=0), vb)
    assert F.array_equal(g.ndata["h"], bg.ndata["h"])
    assert F.array_equal(
        F.cat([g.edata["h"], g.edata["h"]], dim=0), bg.edata["h"]
    )
    bg.ndata["hh"] = F.tensor([[0.0], [1.0], [2.0], [1.0]])
    assert ("hh" in g.ndata) is False
    bg.edata["hh"] = F.tensor(
        [[0.0], [1.0], [2.0], [1.0], [0.0], [1.0], [2.0], [1.0]]
    )
    assert ("hh" in g.edata) is False

    # donot share ndata and edata
    bg = dgl.add_reverse_edges(g, copy_ndata=False, copy_edata=False)
    ub, vb = bg.edges()
    assert F.array_equal(F.cat([u, v], dim=0), ub)
    assert F.array_equal(F.cat([v, u], dim=0), vb)
    assert ("h" in bg.ndata) is False
    assert ("h" in bg.edata) is False

    # zero edge graph
    g = dgl.graph(([], []))
    bg = dgl.add_reverse_edges(
        g, copy_ndata=True, copy_edata=True, exclude_self=False
    )

    # heterogeneous graph
    g = dgl.heterograph(
        {
            ("user", "wins", "user"): (
                F.tensor([0, 2, 0, 2, 2]),
                F.tensor([1, 1, 2, 1, 0]),
            ),
            ("user", "plays", "game"): (
                F.tensor([1, 2, 1]),
                F.tensor([2, 1, 1]),
            ),
            ("user", "follows", "user"): (
                F.tensor([1, 2, 1]),
                F.tensor([0, 0, 0]),
            ),
        }
    )
    g.nodes["game"].data["hv"] = F.ones((3, 1))
    g.nodes["user"].data["hv"] = F.ones((3, 1))
    g.edges["wins"].data["h"] = F.tensor([0, 1, 2, 3, 4])
    bg = dgl.add_reverse_edges(
        g, copy_ndata=True, copy_edata=True, ignore_bipartite=True
    )
    assert F.array_equal(
        g.nodes["game"].data["hv"], bg.nodes["game"].data["hv"]
    )
    assert F.array_equal(
        g.nodes["user"].data["hv"], bg.nodes["user"].data["hv"]
    )
    u, v = g.all_edges(order="eid", etype=("user", "wins", "user"))
    ub, vb = bg.all_edges(order="eid", etype=("user", "wins", "user"))
    assert F.array_equal(F.cat([u, v], dim=0), ub)
    assert F.array_equal(F.cat([v, u], dim=0), vb)
    assert F.array_equal(
        F.cat([g.edges["wins"].data["h"], g.edges["wins"].data["h"]], dim=0),
        bg.edges["wins"].data["h"],
    )
    u, v = g.all_edges(order="eid", etype=("user", "follows", "user"))
    ub, vb = bg.all_edges(order="eid", etype=("user", "follows", "user"))
    assert F.array_equal(F.cat([u, v], dim=0), ub)
    assert F.array_equal(F.cat([v, u], dim=0), vb)
    u, v = g.all_edges(order="eid", etype=("user", "plays", "game"))
    ub, vb = bg.all_edges(order="eid", etype=("user", "plays", "game"))
    assert F.array_equal(u, ub)
    assert F.array_equal(v, vb)
    assert set(bg.edges["plays"].data.keys()) == {dgl.EID}
    assert set(bg.edges["follows"].data.keys()) == {dgl.EID}

    # donot share ndata and edata
    bg = dgl.add_reverse_edges(
        g, copy_ndata=False, copy_edata=False, ignore_bipartite=True
    )
    assert len(bg.edges["wins"].data) == 0
    assert len(bg.edges["plays"].data) == 0
    assert len(bg.edges["follows"].data) == 0
    assert len(bg.nodes["game"].data) == 0
    assert len(bg.nodes["user"].data) == 0
    u, v = g.all_edges(order="eid", etype=("user", "wins", "user"))
    ub, vb = bg.all_edges(order="eid", etype=("user", "wins", "user"))
    assert F.array_equal(F.cat([u, v], dim=0), ub)
    assert F.array_equal(F.cat([v, u], dim=0), vb)
    u, v = g.all_edges(order="eid", etype=("user", "follows", "user"))
    ub, vb = bg.all_edges(order="eid", etype=("user", "follows", "user"))
    assert F.array_equal(F.cat([u, v], dim=0), ub)
    assert F.array_equal(F.cat([v, u], dim=0), vb)
    u, v = g.all_edges(order="eid", etype=("user", "plays", "game"))
    ub, vb = bg.all_edges(order="eid", etype=("user", "plays", "game"))
    assert F.array_equal(u, ub)
    assert F.array_equal(v, vb)

    # test the case when some nodes have zero degree
    # homogeneous graph
    g = dgl.graph((F.tensor([0, 1, 3, 1]), F.tensor([1, 2, 0, 2])), num_nodes=6)
    g.ndata["h"] = F.tensor([[0.0], [1.0], [2.0], [1.0], [1.0], [1.0]])
    g.edata["h"] = F.tensor([[3.0], [4.0], [5.0], [6.0]])
    bg = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
    assert g.num_nodes() == bg.num_nodes()
    assert F.array_equal(g.ndata["h"], bg.ndata["h"])
    assert F.array_equal(
        F.cat([g.edata["h"], g.edata["h"]], dim=0), bg.edata["h"]
    )

    # heterogeneous graph
    g = dgl.heterograph(
        {
            ("user", "wins", "user"): (
                F.tensor([0, 2, 0, 2, 2]),
                F.tensor([1, 1, 2, 1, 0]),
            ),
            ("user", "plays", "game"): (
                F.tensor([1, 2, 1]),
                F.tensor([2, 1, 1]),
            ),
            ("user", "follows", "user"): (
                F.tensor([1, 2, 1]),
                F.tensor([0, 0, 0]),
            ),
        },
        num_nodes_dict={"user": 5, "game": 3},
    )
    g.nodes["game"].data["hv"] = F.ones((3, 1))
    g.nodes["user"].data["hv"] = F.ones((5, 1))
    g.edges["wins"].data["h"] = F.tensor([0, 1, 2, 3, 4])
    bg = dgl.add_reverse_edges(
        g, copy_ndata=True, copy_edata=True, ignore_bipartite=True
    )
    assert g.num_nodes("user") == bg.num_nodes("user")
    assert g.num_nodes("game") == bg.num_nodes("game")
    assert F.array_equal(
        g.nodes["game"].data["hv"], bg.nodes["game"].data["hv"]
    )
    assert F.array_equal(
        g.nodes["user"].data["hv"], bg.nodes["user"].data["hv"]
    )
    assert F.array_equal(
        F.cat([g.edges["wins"].data["h"], g.edges["wins"].data["h"]], dim=0),
        bg.edges["wins"].data["h"],
    )

    # test exclude_self
    g = dgl.heterograph(
        {
            ("A", "r1", "A"): (F.tensor([0, 0, 1, 1]), F.tensor([0, 1, 1, 2])),
            ("A", "r2", "A"): (F.tensor([0, 1]), F.tensor([1, 2])),
        }
    )
    g.edges["r1"].data["h"] = F.tensor([0, 1, 2, 3])
    rg = dgl.add_reverse_edges(g, copy_edata=True, exclude_self=True)
    assert rg.num_edges("r1") == 6
    assert rg.num_edges("r2") == 4
    assert F.array_equal(rg.edges["r1"].data["h"], F.tensor([0, 1, 2, 3, 1, 3]))


@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
def test_simple_graph():
    elist = [(0, 1), (0, 2), (1, 2), (0, 1)]
    g = dgl.graph(elist)
    assert g.is_multigraph
    sg = dgl.to_simple(g)
    assert not sg.is_multigraph
    assert sg.num_edges() == 3
    src, dst = sg.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == set(elist)


@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
def _test_bidirected_graph():
    def _test(in_readonly, out_readonly):
        elist = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2)]
        num_edges = 7
        g = dgl.graph(elist)
        elist.append((1, 2))
        elist = set(elist)
        big = dgl.to_bidirected_stale(g, out_readonly)
        assert big.num_edges() == num_edges
        src, dst = big.edges()
        eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
        assert eset == set(elist)

    _test(True, True)
    _test(True, False)
    _test(False, True)
    _test(False, False)


@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
def test_khop_graph():
    N = 20
    feat = F.randn((N, 5))

    def _test(g):
        for k in range(4):
            g_k = dgl.khop_graph(g, k)
            # use original graph to do message passing for k times.
            g.ndata["h"] = feat
            for _ in range(k):
                g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            h_0 = g.ndata.pop("h")
            # use k-hop graph to do message passing for one time.
            g_k.ndata["h"] = feat
            g_k.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            h_1 = g_k.ndata.pop("h")
            assert F.allclose(h_0, h_1, rtol=1e-3, atol=1e-3)

    # Test for random undirected graphs
    g = dgl.from_networkx(nx.erdos_renyi_graph(N, 0.3))
    _test(g)
    # Test for random directed graphs
    g = dgl.from_networkx(nx.erdos_renyi_graph(N, 0.3, directed=True))
    _test(g)


@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
def test_khop_adj():
    N = 20
    feat = F.randn((N, 5))
    g = dgl.from_networkx(nx.erdos_renyi_graph(N, 0.3, directed=True))
    for k in range(3):
        adj = F.tensor(F.swapaxes(dgl.khop_adj(g, k), 0, 1))
        # use original graph to do message passing for k times.
        g.ndata["h"] = feat
        for _ in range(k):
            g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
        h_0 = g.ndata.pop("h")
        # use k-hop adj to do message passing for one time.
        h_1 = F.matmul(adj, feat)
        assert F.allclose(h_0, h_1, rtol=1e-3, atol=1e-3)


@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
def test_laplacian_lambda_max():
    N = 20
    eps = 1e-6
    # test DGLGraph
    g = dgl.from_networkx(nx.erdos_renyi_graph(N, 0.3))
    l_max = dgl.laplacian_lambda_max(g)
    assert l_max[0] < 2 + eps
    # test batched DGLGraph
    """
    N_arr = [20, 30, 10, 12]
    bg = dgl.batch([
        dgl.from_networkx(nx.erdos_renyi_graph(N, 0.3))
        for N in N_arr
    ])
    l_max_arr = dgl.laplacian_lambda_max(bg)
    assert len(l_max_arr) == len(N_arr)
    for l_max in l_max_arr:
        assert l_max < 2 + eps
    """


def create_large_graph(num_nodes, idtype=F.int64):
    row = np.random.choice(num_nodes, num_nodes * 10)
    col = np.random.choice(num_nodes, num_nodes * 10)
    spm = spsp.coo_matrix((np.ones(len(row)), (row, col)))
    spm.sum_duplicates()

    return dgl.from_scipy(spm, idtype=idtype)


# Disabled since everything will be on heterogeneous graphs
@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
def test_partition_with_halo():
    g = create_large_graph(1000)
    node_part = np.random.choice(4, g.num_nodes())
    subgs, _, _ = dgl.transforms.partition_graph_with_halo(
        g, node_part, 2, reshuffle=True
    )
    for part_id, subg in subgs.items():
        node_ids = np.nonzero(node_part == part_id)[0]
        lnode_ids = np.nonzero(F.asnumpy(subg.ndata["inner_node"]))[0]
        orig_nids = F.asnumpy(subg.ndata["orig_id"])[lnode_ids]
        assert np.all(np.sort(orig_nids) == node_ids)
        assert np.all(
            F.asnumpy(subg.in_degrees(lnode_ids))
            == F.asnumpy(g.in_degrees(orig_nids))
        )
        assert np.all(
            F.asnumpy(subg.out_degrees(lnode_ids))
            == F.asnumpy(g.out_degrees(orig_nids))
        )


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
@unittest.skipIf(
    F._default_context_str == "gpu", reason="METIS doesn't support GPU"
)
@parametrize_idtype
def test_metis_partition(idtype):
    # TODO(zhengda) Metis fails to partition a small graph.
    g = create_large_graph(1000, idtype=idtype)
    if idtype == F.int64:
        check_metis_partition(g, 0)
        check_metis_partition(g, 1)
        check_metis_partition(g, 2)
        check_metis_partition_with_constraint(g)
    else:
        assert_fail = False
        try:
            check_metis_partition(g, 1)
        except:
            assert_fail = True
        assert assert_fail


def check_metis_partition_with_constraint(g):
    ntypes = np.zeros((g.num_nodes(),), dtype=np.int32)
    ntypes[0 : int(g.num_nodes() / 4)] = 1
    ntypes[int(g.num_nodes() * 3 / 4) :] = 2
    subgs = dgl.transforms.metis_partition(
        g, 4, extra_cached_hops=1, balance_ntypes=ntypes
    )
    if subgs is not None:
        for i in subgs:
            subg = subgs[i]
            parent_nids = F.asnumpy(subg.ndata[dgl.NID])
            sub_ntypes = ntypes[parent_nids]
            print("type0:", np.sum(sub_ntypes == 0))
            print("type1:", np.sum(sub_ntypes == 1))
            print("type2:", np.sum(sub_ntypes == 2))
    subgs = dgl.transforms.metis_partition(
        g, 4, extra_cached_hops=1, balance_ntypes=ntypes, balance_edges=True
    )
    if subgs is not None:
        for i in subgs:
            subg = subgs[i]
            parent_nids = F.asnumpy(subg.ndata[dgl.NID])
            sub_ntypes = ntypes[parent_nids]
            print("type0:", np.sum(sub_ntypes == 0))
            print("type1:", np.sum(sub_ntypes == 1))
            print("type2:", np.sum(sub_ntypes == 2))


def check_metis_partition(g, extra_hops):
    subgs = dgl.transforms.metis_partition(g, 4, extra_cached_hops=extra_hops)
    num_inner_nodes = 0
    num_inner_edges = 0
    if subgs is not None:
        for part_id, subg in subgs.items():
            lnode_ids = np.nonzero(F.asnumpy(subg.ndata["inner_node"]))[0]
            ledge_ids = np.nonzero(F.asnumpy(subg.edata["inner_edge"]))[0]
            num_inner_nodes += len(lnode_ids)
            num_inner_edges += len(ledge_ids)
            assert np.sum(F.asnumpy(subg.ndata["part_id"]) == part_id) == len(
                lnode_ids
            )
        assert num_inner_nodes == g.num_nodes()
        print(g.num_edges() - num_inner_edges)

    if extra_hops == 0:
        return

    # partitions with node reshuffling
    subgs = dgl.transforms.metis_partition(
        g, 4, extra_cached_hops=extra_hops, reshuffle=True
    )
    num_inner_nodes = 0
    num_inner_edges = 0
    edge_cnts = np.zeros((g.num_edges(),))
    if subgs is not None:
        for part_id, subg in subgs.items():
            lnode_ids = np.nonzero(F.asnumpy(subg.ndata["inner_node"]))[0]
            ledge_ids = np.nonzero(F.asnumpy(subg.edata["inner_edge"]))[0]
            num_inner_nodes += len(lnode_ids)
            num_inner_edges += len(ledge_ids)
            assert np.sum(F.asnumpy(subg.ndata["part_id"]) == part_id) == len(
                lnode_ids
            )
            nids = F.asnumpy(subg.ndata[dgl.NID])

            # ensure the local node Ids are contiguous.
            parent_ids = F.asnumpy(subg.ndata[dgl.NID])
            parent_ids = parent_ids[: len(lnode_ids)]
            assert np.all(
                parent_ids == np.arange(parent_ids[0], parent_ids[-1] + 1)
            )

            # count the local edges.
            parent_ids = F.asnumpy(subg.edata[dgl.EID])[ledge_ids]
            edge_cnts[parent_ids] += 1

            orig_ids = subg.ndata["orig_id"]
            inner_node = F.asnumpy(subg.ndata["inner_node"])
            for nid in range(subg.num_nodes()):
                neighs = subg.predecessors(nid)
                old_neighs1 = F.gather_row(orig_ids, neighs)
                old_nid = F.asnumpy(orig_ids[nid])
                old_neighs2 = g.predecessors(old_nid)
                # If this is an inner node, it should have the full neighborhood.
                if inner_node[nid]:
                    assert np.all(
                        np.sort(F.asnumpy(old_neighs1))
                        == np.sort(F.asnumpy(old_neighs2))
                    )
        # Normally, local edges are only counted once.
        assert np.all(edge_cnts == 1)

        assert num_inner_nodes == g.num_nodes()
        print(g.num_edges() - num_inner_edges)


@unittest.skipIf(
    F._default_context_str == "gpu", reason="It doesn't support GPU"
)
def test_reorder_nodes():
    g = create_large_graph(1000)
    new_nids = np.random.permutation(g.num_nodes())
    # TODO(zhengda) we need to test both CSR and COO.
    new_g = dgl.partition.reorder_nodes(g, new_nids)
    new_in_deg = new_g.in_degrees()
    new_out_deg = new_g.out_degrees()
    in_deg = g.in_degrees()
    out_deg = g.out_degrees()
    new_in_deg1 = F.scatter_row(in_deg, F.tensor(new_nids), in_deg)
    new_out_deg1 = F.scatter_row(out_deg, F.tensor(new_nids), out_deg)
    assert np.all(F.asnumpy(new_in_deg == new_in_deg1))
    assert np.all(F.asnumpy(new_out_deg == new_out_deg1))
    orig_ids = F.asnumpy(new_g.ndata["orig_id"])
    for nid in range(g.num_nodes()):
        neighs = F.asnumpy(g.successors(nid))
        new_neighs1 = new_nids[neighs]
        new_nid = new_nids[nid]
        new_neighs2 = new_g.successors(new_nid)
        assert np.all(np.sort(new_neighs1) == np.sort(F.asnumpy(new_neighs2)))

    for nid in range(new_g.num_nodes()):
        neighs = F.asnumpy(new_g.successors(nid))
        old_neighs1 = orig_ids[neighs]
        old_nid = orig_ids[nid]
        old_neighs2 = g.successors(old_nid)
        assert np.all(np.sort(old_neighs1) == np.sort(F.asnumpy(old_neighs2)))

        neighs = F.asnumpy(new_g.predecessors(nid))
        old_neighs1 = orig_ids[neighs]
        old_nid = orig_ids[nid]
        old_neighs2 = g.predecessors(old_nid)
        assert np.all(np.sort(old_neighs1) == np.sort(F.asnumpy(old_neighs2)))


@parametrize_idtype
def test_compact(idtype):
    g1 = dgl.heterograph(
        {
            ("user", "follow", "user"): ([1, 3], [3, 5]),
            ("user", "plays", "game"): ([2, 3, 2], [4, 4, 5]),
            ("game", "wished-by", "user"): ([6, 5], [7, 7]),
        },
        {"user": 20, "game": 10},
        idtype=idtype,
        device=F.ctx(),
    )

    g2 = dgl.heterograph(
        {
            ("game", "clicked-by", "user"): ([3], [1]),
            ("user", "likes", "user"): ([1, 8], [8, 9]),
        },
        {"user": 20, "game": 10},
        idtype=idtype,
        device=F.ctx(),
    )

    g3 = dgl.heterograph(
        {("user", "_E", "user"): ((0, 1), (1, 2))},
        {"user": 10},
        idtype=idtype,
        device=F.ctx(),
    )
    g4 = dgl.heterograph(
        {("user", "_E", "user"): ((1, 3), (3, 5))},
        {"user": 10},
        idtype=idtype,
        device=F.ctx(),
    )

    def _check(g, new_g, induced_nodes):
        assert g.ntypes == new_g.ntypes
        assert g.canonical_etypes == new_g.canonical_etypes

        for ntype in g.ntypes:
            assert -1 not in induced_nodes[ntype]

        for etype in g.canonical_etypes:
            g_src, g_dst = g.all_edges(order="eid", etype=etype)
            g_src = F.asnumpy(g_src)
            g_dst = F.asnumpy(g_dst)
            new_g_src, new_g_dst = new_g.all_edges(order="eid", etype=etype)
            new_g_src_mapped = induced_nodes[etype[0]][F.asnumpy(new_g_src)]
            new_g_dst_mapped = induced_nodes[etype[2]][F.asnumpy(new_g_dst)]
            assert (g_src == new_g_src_mapped).all()
            assert (g_dst == new_g_dst_mapped).all()

    # Test default
    new_g1 = dgl.compact_graphs(g1)
    induced_nodes = {
        ntype: new_g1.nodes[ntype].data[dgl.NID] for ntype in new_g1.ntypes
    }
    induced_nodes = {k: F.asnumpy(v) for k, v in induced_nodes.items()}
    assert new_g1.idtype == idtype
    assert set(induced_nodes["user"]) == set([1, 3, 5, 2, 7])
    assert set(induced_nodes["game"]) == set([4, 5, 6])
    _check(g1, new_g1, induced_nodes)

    # Test with always_preserve given a dict
    new_g1 = dgl.compact_graphs(
        g1, always_preserve={"game": F.tensor([4, 7], idtype)}
    )
    assert new_g1.idtype == idtype
    induced_nodes = {
        ntype: new_g1.nodes[ntype].data[dgl.NID] for ntype in new_g1.ntypes
    }
    induced_nodes = {k: F.asnumpy(v) for k, v in induced_nodes.items()}
    assert set(induced_nodes["user"]) == set([1, 3, 5, 2, 7])
    assert set(induced_nodes["game"]) == set([4, 5, 6, 7])
    _check(g1, new_g1, induced_nodes)

    # Test with always_preserve given a tensor
    new_g3 = dgl.compact_graphs(g3, always_preserve=F.tensor([1, 7], idtype))
    induced_nodes = {
        ntype: new_g3.nodes[ntype].data[dgl.NID] for ntype in new_g3.ntypes
    }
    induced_nodes = {k: F.asnumpy(v) for k, v in induced_nodes.items()}

    assert new_g3.idtype == idtype
    assert set(induced_nodes["user"]) == set([0, 1, 2, 7])
    _check(g3, new_g3, induced_nodes)

    # Test multiple graphs
    new_g1, new_g2 = dgl.compact_graphs([g1, g2])
    induced_nodes = {
        ntype: new_g1.nodes[ntype].data[dgl.NID] for ntype in new_g1.ntypes
    }
    induced_nodes = {k: F.asnumpy(v) for k, v in induced_nodes.items()}
    assert new_g1.idtype == idtype
    assert new_g2.idtype == idtype
    assert set(induced_nodes["user"]) == set([1, 3, 5, 2, 7, 8, 9])
    assert set(induced_nodes["game"]) == set([3, 4, 5, 6])
    _check(g1, new_g1, induced_nodes)
    _check(g2, new_g2, induced_nodes)

    # Test multiple graphs with always_preserve given a dict
    new_g1, new_g2 = dgl.compact_graphs(
        [g1, g2], always_preserve={"game": F.tensor([4, 7], dtype=idtype)}
    )
    induced_nodes = {
        ntype: new_g1.nodes[ntype].data[dgl.NID] for ntype in new_g1.ntypes
    }
    induced_nodes = {k: F.asnumpy(v) for k, v in induced_nodes.items()}
    assert new_g1.idtype == idtype
    assert new_g2.idtype == idtype
    assert set(induced_nodes["user"]) == set([1, 3, 5, 2, 7, 8, 9])
    assert set(induced_nodes["game"]) == set([3, 4, 5, 6, 7])
    _check(g1, new_g1, induced_nodes)
    _check(g2, new_g2, induced_nodes)

    # Test multiple graphs with always_preserve given a tensor
    new_g3, new_g4 = dgl.compact_graphs(
        [g3, g4], always_preserve=F.tensor([1, 7], dtype=idtype)
    )
    induced_nodes = {
        ntype: new_g3.nodes[ntype].data[dgl.NID] for ntype in new_g3.ntypes
    }
    induced_nodes = {k: F.asnumpy(v) for k, v in induced_nodes.items()}

    assert new_g3.idtype == idtype
    assert new_g4.idtype == idtype

    assert set(induced_nodes["user"]) == set([0, 1, 2, 3, 5, 7])
    _check(g3, new_g3, induced_nodes)
    _check(g4, new_g4, induced_nodes)


@unittest.skipIf(
    F._default_context_str == "gpu", reason="GPU to simple not implemented"
)
@parametrize_idtype
def test_to_simple(idtype):
    # homogeneous graph
    g = dgl.graph((F.tensor([0, 1, 2, 1]), F.tensor([1, 2, 0, 2])))
    g.ndata["h"] = F.tensor([[0.0], [1.0], [2.0]])
    g.edata["h"] = F.tensor([[3.0], [4.0], [5.0], [6.0]])
    sg, wb = dgl.to_simple(g, writeback_mapping=True)
    u, v = g.all_edges(form="uv", order="eid")
    u = F.asnumpy(u).tolist()
    v = F.asnumpy(v).tolist()
    uv = list(zip(u, v))
    eid_map = F.asnumpy(wb)

    su, sv = sg.all_edges(form="uv", order="eid")
    su = F.asnumpy(su).tolist()
    sv = F.asnumpy(sv).tolist()
    suv = list(zip(su, sv))
    sc = F.asnumpy(sg.edata["count"])
    assert set(uv) == set(suv)
    for i, e in enumerate(suv):
        assert sc[i] == sum(e == _e for _e in uv)
    for i, e in enumerate(uv):
        assert eid_map[i] == suv.index(e)
    # shared ndata
    assert F.array_equal(sg.ndata["h"], g.ndata["h"])
    assert "h" not in sg.edata
    # new ndata to sg
    sg.ndata["hh"] = F.tensor([[0.0], [1.0], [2.0]])
    assert "hh" not in g.ndata

    sg = dgl.to_simple(g, writeback_mapping=False, copy_ndata=False)
    assert "h" not in sg.ndata
    assert "h" not in sg.edata

    # test coalesce edge feature
    sg = dgl.to_simple(g, copy_edata=True, aggregator="arbitrary")
    assert F.allclose(sg.edata["h"][1], F.tensor([4.0]))
    sg = dgl.to_simple(g, copy_edata=True, aggregator="sum")
    assert F.allclose(sg.edata["h"][1], F.tensor([10.0]))
    sg = dgl.to_simple(g, copy_edata=True, aggregator="mean")
    assert F.allclose(sg.edata["h"][1], F.tensor([5.0]))

    # heterogeneous graph
    g = dgl.heterograph(
        {
            ("user", "follow", "user"): (
                [0, 1, 2, 1, 1, 1],
                [1, 3, 2, 3, 4, 4],
            ),
            ("user", "plays", "game"): (
                [3, 2, 1, 1, 3, 2, 2],
                [5, 3, 4, 4, 5, 3, 3],
            ),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    g.nodes["user"].data["h"] = F.tensor([0, 1, 2, 3, 4])
    g.nodes["user"].data["hh"] = F.tensor([0, 1, 2, 3, 4])
    g.edges["follow"].data["h"] = F.tensor([0, 1, 2, 3, 4, 5])
    sg, wb = dgl.to_simple(
        g, return_counts="weights", writeback_mapping=True, copy_edata=True
    )
    g.nodes["game"].data["h"] = F.tensor([0, 1, 2, 3, 4, 5])

    for etype in g.canonical_etypes:
        u, v = g.all_edges(form="uv", order="eid", etype=etype)
        u = F.asnumpy(u).tolist()
        v = F.asnumpy(v).tolist()
        uv = list(zip(u, v))
        eid_map = F.asnumpy(wb[etype])

        su, sv = sg.all_edges(form="uv", order="eid", etype=etype)
        su = F.asnumpy(su).tolist()
        sv = F.asnumpy(sv).tolist()
        suv = list(zip(su, sv))
        sw = F.asnumpy(sg.edges[etype].data["weights"])

        assert set(uv) == set(suv)
        for i, e in enumerate(suv):
            assert sw[i] == sum(e == _e for _e in uv)
        for i, e in enumerate(uv):
            assert eid_map[i] == suv.index(e)
    # shared ndata
    assert F.array_equal(sg.nodes["user"].data["h"], g.nodes["user"].data["h"])
    assert F.array_equal(
        sg.nodes["user"].data["hh"], g.nodes["user"].data["hh"]
    )
    assert "h" not in sg.nodes["game"].data
    # new ndata to sg
    sg.nodes["user"].data["hhh"] = F.tensor([0, 1, 2, 3, 4])
    assert "hhh" not in g.nodes["user"].data
    # share edata
    feat_idx = F.asnumpy(wb[("user", "follow", "user")])
    _, indices = np.unique(feat_idx, return_index=True)
    assert np.array_equal(
        F.asnumpy(sg.edges["follow"].data["h"]),
        F.asnumpy(g.edges["follow"].data["h"])[indices],
    )

    sg = dgl.to_simple(g, writeback_mapping=False, copy_ndata=False)
    for ntype in g.ntypes:
        assert g.num_nodes(ntype) == sg.num_nodes(ntype)
    assert "h" not in sg.nodes["user"].data
    assert "hh" not in sg.nodes["user"].data

    # verify DGLGraph.edge_ids() after dgl.to_simple()
    # in case ids are not initialized in underlying coo2csr()
    u = F.tensor([0, 1, 2])
    v = F.tensor([1, 2, 3])
    eids = F.tensor([0, 1, 2])
    g = dgl.graph((u, v))
    assert F.array_equal(g.edge_ids(u, v), eids)
    sg = dgl.to_simple(g)
    assert F.array_equal(sg.edge_ids(u, v), eids)


@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
@parametrize_idtype
def test_remove_edges(idtype):
    def check(g1, etype, g, edges_removed):
        src, dst, eid = g.edges(etype=etype, form="all")
        src1, dst1 = g1.edges(etype=etype, order="eid")
        if etype is not None:
            eid1 = g1.edges[etype].data[dgl.EID]
        else:
            eid1 = g1.edata[dgl.EID]
        src1 = F.asnumpy(src1)
        dst1 = F.asnumpy(dst1)
        eid1 = F.asnumpy(eid1)
        src = F.asnumpy(src)
        dst = F.asnumpy(dst)
        eid = F.asnumpy(eid)
        sde_set = set(zip(src, dst, eid))

        for s, d, e in zip(src1, dst1, eid1):
            assert (s, d, e) in sde_set
        assert not np.isin(edges_removed, eid1).any()
        assert g1.idtype == g.idtype

    for fmt in ["coo", "csr", "csc"]:
        for edges_to_remove in [[2], [2, 2], [3, 2], [1, 3, 1, 2]]:
            g = dgl.graph(([0, 2, 1, 3], [1, 3, 2, 4]), idtype=idtype).formats(
                fmt
            )
            g1 = dgl.remove_edges(g, F.tensor(edges_to_remove, idtype))
            check(g1, None, g, edges_to_remove)

            g = dgl.from_scipy(
                spsp.csr_matrix(
                    ([1, 1, 1, 1], ([0, 2, 1, 3], [1, 3, 2, 4])), shape=(5, 5)
                ),
                idtype=idtype,
            ).formats(fmt)
            g1 = dgl.remove_edges(g, F.tensor(edges_to_remove, idtype))
            check(g1, None, g, edges_to_remove)

    g = dgl.heterograph(
        {
            ("A", "AA", "A"): ([0, 2, 1, 3], [1, 3, 2, 4]),
            ("A", "AB", "B"): ([0, 1, 3, 1], [1, 3, 5, 6]),
            ("B", "BA", "A"): ([2, 3], [3, 2]),
        },
        idtype=idtype,
    )
    g2 = dgl.remove_edges(
        g,
        {
            "AA": F.tensor([2], idtype),
            "AB": F.tensor([3], idtype),
            "BA": F.tensor([1], idtype),
        },
    )
    check(g2, "AA", g, [2])
    check(g2, "AB", g, [3])
    check(g2, "BA", g, [1])

    g3 = dgl.remove_edges(
        g,
        {
            "AA": F.tensor([], idtype),
            "AB": F.tensor([3], idtype),
            "BA": F.tensor([1], idtype),
        },
    )
    check(g3, "AA", g, [])
    check(g3, "AB", g, [3])
    check(g3, "BA", g, [1])

    g4 = dgl.remove_edges(g, {"AB": F.tensor([3, 1, 2, 0], idtype)})
    check(g4, "AA", g, [])
    check(g4, "AB", g, [3, 1, 2, 0])
    check(g4, "BA", g, [])


@parametrize_idtype
def test_add_edges(idtype):
    # homogeneous graph
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    u = 0
    v = 1
    g = dgl.add_edges(g, u, v)
    assert g.device == F.ctx()
    assert g.num_nodes() == 3
    assert g.num_edges() == 3
    u = [0]
    v = [1]
    g = dgl.add_edges(g, u, v)
    assert g.device == F.ctx()
    assert g.num_nodes() == 3
    assert g.num_edges() == 4
    u = F.tensor(u, dtype=idtype)
    v = F.tensor(v, dtype=idtype)
    g = dgl.add_edges(g, u, v)
    assert g.device == F.ctx()
    assert g.num_nodes() == 3
    assert g.num_edges() == 5
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([0, 1, 0, 0, 0], dtype=idtype))
    assert F.array_equal(v, F.tensor([1, 2, 1, 1, 1], dtype=idtype))
    g = dgl.add_edges(g, [], [])
    g = dgl.add_edges(g, 0, [])
    g = dgl.add_edges(g, [], 0)
    assert g.device == F.ctx()
    assert g.num_nodes() == 3
    assert g.num_edges() == 5
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([0, 1, 0, 0, 0], dtype=idtype))
    assert F.array_equal(v, F.tensor([1, 2, 1, 1, 1], dtype=idtype))

    # node id larger than current max node id
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    u = F.tensor([0, 1], dtype=idtype)
    v = F.tensor([2, 3], dtype=idtype)
    g = dgl.add_edges(g, u, v)
    assert g.num_nodes() == 4
    assert g.num_edges() == 4
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([0, 1, 0, 1], dtype=idtype))
    assert F.array_equal(v, F.tensor([1, 2, 2, 3], dtype=idtype))

    # has data
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    g.ndata["h"] = F.copy_to(F.tensor([1, 1, 1], dtype=idtype), ctx=F.ctx())
    g.edata["h"] = F.copy_to(F.tensor([1, 1], dtype=idtype), ctx=F.ctx())
    u = F.tensor([0, 1], dtype=idtype)
    v = F.tensor([2, 3], dtype=idtype)
    e_feat = {
        "h": F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx()),
        "hh": F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx()),
    }
    g = dgl.add_edges(g, u, v, e_feat)
    assert g.num_nodes() == 4
    assert g.num_edges() == 4
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([0, 1, 0, 1], dtype=idtype))
    assert F.array_equal(v, F.tensor([1, 2, 2, 3], dtype=idtype))
    assert F.array_equal(g.ndata["h"], F.tensor([1, 1, 1, 0], dtype=idtype))
    assert F.array_equal(g.edata["h"], F.tensor([1, 1, 2, 2], dtype=idtype))
    assert F.array_equal(g.edata["hh"], F.tensor([0, 0, 2, 2], dtype=idtype))

    # zero data graph
    g = dgl.graph(([], []), num_nodes=0, idtype=idtype, device=F.ctx())
    u = F.tensor([0, 1], dtype=idtype)
    v = F.tensor([2, 2], dtype=idtype)
    e_feat = {
        "h": F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx()),
        "hh": F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx()),
    }
    g = dgl.add_edges(g, u, v, e_feat)
    assert g.num_nodes() == 3
    assert g.num_edges() == 2
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([0, 1], dtype=idtype))
    assert F.array_equal(v, F.tensor([2, 2], dtype=idtype))
    assert F.array_equal(g.edata["h"], F.tensor([2, 2], dtype=idtype))
    assert F.array_equal(g.edata["hh"], F.tensor([2, 2], dtype=idtype))

    # bipartite graph
    g = dgl.heterograph(
        {("user", "plays", "game"): ([0, 1], [1, 2])},
        idtype=idtype,
        device=F.ctx(),
    )
    u = 0
    v = 1
    g = dgl.add_edges(g, u, v)
    assert g.device == F.ctx()
    assert g.num_nodes("user") == 2
    assert g.num_nodes("game") == 3
    assert g.num_edges() == 3
    u = [0]
    v = [1]
    g = dgl.add_edges(g, u, v)
    assert g.device == F.ctx()
    assert g.num_nodes("user") == 2
    assert g.num_nodes("game") == 3
    assert g.num_edges() == 4
    u = F.tensor(u, dtype=idtype)
    v = F.tensor(v, dtype=idtype)
    g = dgl.add_edges(g, u, v)
    assert g.device == F.ctx()
    assert g.num_nodes("user") == 2
    assert g.num_nodes("game") == 3
    assert g.num_edges() == 5
    u, v = g.edges(form="uv")
    assert F.array_equal(u, F.tensor([0, 1, 0, 0, 0], dtype=idtype))
    assert F.array_equal(v, F.tensor([1, 2, 1, 1, 1], dtype=idtype))

    # node id larger than current max node id
    g = dgl.heterograph(
        {("user", "plays", "game"): ([0, 1], [1, 2])},
        idtype=idtype,
        device=F.ctx(),
    )
    u = F.tensor([0, 2], dtype=idtype)
    v = F.tensor([2, 3], dtype=idtype)
    g = dgl.add_edges(g, u, v)
    assert g.device == F.ctx()
    assert g.num_nodes("user") == 3
    assert g.num_nodes("game") == 4
    assert g.num_edges() == 4
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([0, 1, 0, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([1, 2, 2, 3], dtype=idtype))

    # has data
    g = dgl.heterograph(
        {("user", "plays", "game"): ([0, 1], [1, 2])},
        idtype=idtype,
        device=F.ctx(),
    )
    g.nodes["user"].data["h"] = F.copy_to(
        F.tensor([1, 1], dtype=idtype), ctx=F.ctx()
    )
    g.nodes["game"].data["h"] = F.copy_to(
        F.tensor([2, 2, 2], dtype=idtype), ctx=F.ctx()
    )
    g.edata["h"] = F.copy_to(F.tensor([1, 1], dtype=idtype), ctx=F.ctx())
    u = F.tensor([0, 2], dtype=idtype)
    v = F.tensor([2, 3], dtype=idtype)
    e_feat = {
        "h": F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx()),
        "hh": F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx()),
    }
    g = dgl.add_edges(g, u, v, e_feat)
    assert g.num_nodes("user") == 3
    assert g.num_nodes("game") == 4
    assert g.num_edges() == 4
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([0, 1, 0, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([1, 2, 2, 3], dtype=idtype))
    assert F.array_equal(
        g.nodes["user"].data["h"], F.tensor([1, 1, 0], dtype=idtype)
    )
    assert F.array_equal(
        g.nodes["game"].data["h"], F.tensor([2, 2, 2, 0], dtype=idtype)
    )
    assert F.array_equal(g.edata["h"], F.tensor([1, 1, 2, 2], dtype=idtype))
    assert F.array_equal(g.edata["hh"], F.tensor([0, 0, 2, 2], dtype=idtype))

    # heterogeneous graph
    g = create_test_heterograph3(idtype)
    u = F.tensor([0, 2], dtype=idtype)
    v = F.tensor([2, 3], dtype=idtype)
    g = dgl.add_edges(g, u, v, etype="plays")
    assert g.num_nodes("user") == 3
    assert g.num_nodes("game") == 4
    assert g.num_nodes("developer") == 2
    assert g.num_edges("plays") == 6
    assert g.num_edges("develops") == 2
    u, v = g.edges(form="uv", order="eid", etype="plays")
    assert F.array_equal(u, F.tensor([0, 1, 1, 2, 0, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([0, 0, 1, 1, 2, 3], dtype=idtype))
    assert F.array_equal(
        g.nodes["user"].data["h"], F.tensor([1, 1, 1], dtype=idtype)
    )
    assert F.array_equal(
        g.nodes["game"].data["h"], F.tensor([2, 2, 0, 0], dtype=idtype)
    )
    assert F.array_equal(
        g.edges["plays"].data["h"], F.tensor([1, 1, 1, 1, 0, 0], dtype=idtype)
    )

    # add with feature
    e_feat = {"h": F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx())}
    u = F.tensor([0, 2], dtype=idtype)
    v = F.tensor([2, 3], dtype=idtype)
    g.nodes["game"].data["h"] = F.copy_to(
        F.tensor([2, 2, 1, 1], dtype=idtype), ctx=F.ctx()
    )
    g = dgl.add_edges(g, u, v, data=e_feat, etype="develops")
    assert g.num_nodes("user") == 3
    assert g.num_nodes("game") == 4
    assert g.num_nodes("developer") == 3
    assert g.num_edges("plays") == 6
    assert g.num_edges("develops") == 4
    u, v = g.edges(form="uv", order="eid", etype="develops")
    assert F.array_equal(u, F.tensor([0, 1, 0, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([0, 1, 2, 3], dtype=idtype))
    assert F.array_equal(
        g.nodes["developer"].data["h"], F.tensor([3, 3, 0], dtype=idtype)
    )
    assert F.array_equal(
        g.nodes["game"].data["h"], F.tensor([2, 2, 1, 1], dtype=idtype)
    )
    assert F.array_equal(
        g.edges["develops"].data["h"], F.tensor([0, 0, 2, 2], dtype=idtype)
    )


@parametrize_idtype
def test_add_nodes(idtype):
    # homogeneous Graphs
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    g.ndata["h"] = F.copy_to(F.tensor([1, 1, 1], dtype=idtype), ctx=F.ctx())
    new_g = dgl.add_nodes(g, 1)
    assert g.num_nodes() == 3
    assert new_g.num_nodes() == 4
    assert F.array_equal(new_g.ndata["h"], F.tensor([1, 1, 1, 0], dtype=idtype))

    # zero node graph
    g = dgl.graph(([], []), num_nodes=3, idtype=idtype, device=F.ctx())
    g.ndata["h"] = F.copy_to(F.tensor([1, 1, 1], dtype=idtype), ctx=F.ctx())
    g = dgl.add_nodes(
        g, 1, data={"h": F.copy_to(F.tensor([2], dtype=idtype), ctx=F.ctx())}
    )
    assert g.num_nodes() == 4
    assert F.array_equal(g.ndata["h"], F.tensor([1, 1, 1, 2], dtype=idtype))

    # bipartite graph
    g = dgl.heterograph(
        {("user", "plays", "game"): ([0, 1], [1, 2])},
        idtype=idtype,
        device=F.ctx(),
    )
    g = dgl.add_nodes(
        g,
        2,
        data={"h": F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx())},
        ntype="user",
    )
    assert g.num_nodes("user") == 4
    assert g.num_nodes("game") == 3
    assert F.array_equal(
        g.nodes["user"].data["h"], F.tensor([0, 0, 2, 2], dtype=idtype)
    )
    g = dgl.add_nodes(g, 2, ntype="game")
    assert g.num_nodes("user") == 4
    assert g.num_nodes("game") == 5

    # heterogeneous graph
    g = create_test_heterograph3(idtype)
    g = dgl.add_nodes(g, 1, ntype="user")
    g = dgl.add_nodes(
        g,
        2,
        data={"h": F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx())},
        ntype="game",
    )
    assert g.num_nodes("user") == 4
    assert g.num_nodes("game") == 4
    assert g.num_nodes("developer") == 2
    assert F.array_equal(
        g.nodes["user"].data["h"], F.tensor([1, 1, 1, 0], dtype=idtype)
    )
    assert F.array_equal(
        g.nodes["game"].data["h"], F.tensor([2, 2, 2, 2], dtype=idtype)
    )


@parametrize_idtype
def test_remove_edges(idtype):
    # homogeneous Graphs
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    e = 0
    g = dgl.remove_edges(g, e)
    assert g.num_edges() == 1
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([1], dtype=idtype))
    assert F.array_equal(v, F.tensor([2], dtype=idtype))
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    e = [0]
    g = dgl.remove_edges(g, e)
    assert g.num_edges() == 1
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([1], dtype=idtype))
    assert F.array_equal(v, F.tensor([2], dtype=idtype))
    e = F.tensor([0], dtype=idtype)
    g = dgl.remove_edges(g, e)
    assert g.num_edges() == 0

    # has node data
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    g.ndata["h"] = F.copy_to(F.tensor([1, 2, 3], dtype=idtype), ctx=F.ctx())
    g = dgl.remove_edges(g, 1)
    assert g.num_edges() == 1
    assert F.array_equal(g.ndata["h"], F.tensor([1, 2, 3], dtype=idtype))

    # has edge data
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    g.edata["h"] = F.copy_to(F.tensor([1, 2], dtype=idtype), ctx=F.ctx())
    g = dgl.remove_edges(g, 0)
    assert g.num_edges() == 1
    assert F.array_equal(g.edata["h"], F.tensor([2], dtype=idtype))

    # invalid eid
    assert_fail = False
    try:
        g = dgl.remove_edges(g, 1)
    except:
        assert_fail = True
    assert assert_fail

    # bipartite graph
    g = dgl.heterograph(
        {("user", "plays", "game"): ([0, 1], [1, 2])},
        idtype=idtype,
        device=F.ctx(),
    )
    e = 0
    g = dgl.remove_edges(g, e)
    assert g.num_edges() == 1
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([1], dtype=idtype))
    assert F.array_equal(v, F.tensor([2], dtype=idtype))
    g = dgl.heterograph(
        {("user", "plays", "game"): ([0, 1], [1, 2])},
        idtype=idtype,
        device=F.ctx(),
    )
    e = [0]
    g = dgl.remove_edges(g, e)
    assert g.num_edges() == 1
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([1], dtype=idtype))
    assert F.array_equal(v, F.tensor([2], dtype=idtype))
    e = F.tensor([0], dtype=idtype)
    g = dgl.remove_edges(g, e)
    assert g.num_edges() == 0

    # has data
    g = dgl.heterograph(
        {("user", "plays", "game"): ([0, 1], [1, 2])},
        idtype=idtype,
        device=F.ctx(),
    )
    g.nodes["user"].data["h"] = F.copy_to(
        F.tensor([1, 1], dtype=idtype), ctx=F.ctx()
    )
    g.nodes["game"].data["h"] = F.copy_to(
        F.tensor([2, 2, 2], dtype=idtype), ctx=F.ctx()
    )
    g.edata["h"] = F.copy_to(F.tensor([1, 2], dtype=idtype), ctx=F.ctx())
    g = dgl.remove_edges(g, 1)
    assert g.num_edges() == 1
    assert F.array_equal(
        g.nodes["user"].data["h"], F.tensor([1, 1], dtype=idtype)
    )
    assert F.array_equal(
        g.nodes["game"].data["h"], F.tensor([2, 2, 2], dtype=idtype)
    )
    assert F.array_equal(g.edata["h"], F.tensor([1], dtype=idtype))

    # heterogeneous graph
    g = create_test_heterograph3(idtype)
    g.edges["plays"].data["h"] = F.copy_to(
        F.tensor([1, 2, 3, 4], dtype=idtype), ctx=F.ctx()
    )
    g = dgl.remove_edges(g, 1, etype="plays")
    assert g.num_edges("plays") == 3
    u, v = g.edges(form="uv", order="eid", etype="plays")
    assert F.array_equal(u, F.tensor([0, 1, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([0, 1, 1], dtype=idtype))
    assert F.array_equal(
        g.edges["plays"].data["h"], F.tensor([1, 3, 4], dtype=idtype)
    )
    # remove all edges of 'develops'
    g = dgl.remove_edges(g, [0, 1], etype="develops")
    assert g.num_edges("develops") == 0
    assert F.array_equal(
        g.nodes["user"].data["h"], F.tensor([1, 1, 1], dtype=idtype)
    )
    assert F.array_equal(
        g.nodes["game"].data["h"], F.tensor([2, 2], dtype=idtype)
    )
    assert F.array_equal(
        g.nodes["developer"].data["h"], F.tensor([3, 3], dtype=idtype)
    )

    # batched graph
    ctx = F.ctx()
    g1 = dgl.graph(([0, 1], [1, 2]), num_nodes=5, idtype=idtype, device=ctx)
    g2 = dgl.graph(([], []), idtype=idtype, device=ctx)
    g3 = dgl.graph(([2, 3, 4], [3, 2, 1]), idtype=idtype, device=ctx)
    bg = dgl.batch([g1, g2, g3])
    bg_r = dgl.remove_edges(bg, 2)
    assert bg.batch_size == bg_r.batch_size
    assert F.array_equal(bg.batch_num_nodes(), bg_r.batch_num_nodes())
    assert F.array_equal(
        bg_r.batch_num_edges(), F.tensor([2, 0, 2], dtype=idtype)
    )

    bg_r = dgl.remove_edges(bg, [0, 2])
    assert bg.batch_size == bg_r.batch_size
    assert F.array_equal(bg.batch_num_nodes(), bg_r.batch_num_nodes())
    assert F.array_equal(
        bg_r.batch_num_edges(), F.tensor([1, 0, 2], dtype=idtype)
    )

    bg_r = dgl.remove_edges(bg, F.tensor([0, 2], dtype=idtype))
    assert bg.batch_size == bg_r.batch_size
    assert F.array_equal(bg.batch_num_nodes(), bg_r.batch_num_nodes())
    assert F.array_equal(
        bg_r.batch_num_edges(), F.tensor([1, 0, 2], dtype=idtype)
    )

    # batched heterogeneous graph
    g1 = dgl.heterograph(
        {
            ("user", "follows", "user"): ([0, 1], [1, 2]),
            ("user", "plays", "game"): ([1, 3], [0, 1]),
        },
        num_nodes_dict={"user": 4, "game": 3},
        idtype=idtype,
        device=ctx,
    )
    g2 = dgl.heterograph(
        {
            ("user", "follows", "user"): ([0, 2], [3, 4]),
            ("user", "plays", "game"): ([], []),
        },
        num_nodes_dict={"user": 6, "game": 2},
        idtype=idtype,
        device=ctx,
    )
    g3 = dgl.heterograph(
        {
            ("user", "follows", "user"): ([], []),
            ("user", "plays", "game"): ([1, 2], [1, 2]),
        },
        idtype=idtype,
        device=ctx,
    )
    bg = dgl.batch([g1, g2, g3])
    bg_r = dgl.remove_edges(bg, 1, etype="follows")
    assert bg.batch_size == bg_r.batch_size
    ntypes = bg.ntypes
    for nty in ntypes:
        assert F.array_equal(bg.batch_num_nodes(nty), bg_r.batch_num_nodes(nty))
    assert F.array_equal(
        bg_r.batch_num_edges("follows"), F.tensor([1, 2, 0], dtype=idtype)
    )
    assert F.array_equal(
        bg_r.batch_num_edges("plays"), bg.batch_num_edges("plays")
    )

    bg_r = dgl.remove_edges(bg, 2, etype="plays")
    assert bg.batch_size == bg_r.batch_size
    for nty in ntypes:
        assert F.array_equal(bg.batch_num_nodes(nty), bg_r.batch_num_nodes(nty))
    assert F.array_equal(
        bg.batch_num_edges("follows"), bg_r.batch_num_edges("follows")
    )
    assert F.array_equal(
        bg_r.batch_num_edges("plays"), F.tensor([2, 0, 1], dtype=idtype)
    )

    bg_r = dgl.remove_edges(bg, [0, 1, 3], etype="follows")
    assert bg.batch_size == bg_r.batch_size
    for nty in ntypes:
        assert F.array_equal(bg.batch_num_nodes(nty), bg_r.batch_num_nodes(nty))
    assert F.array_equal(
        bg_r.batch_num_edges("follows"), F.tensor([0, 1, 0], dtype=idtype)
    )
    assert F.array_equal(
        bg.batch_num_edges("plays"), bg_r.batch_num_edges("plays")
    )

    bg_r = dgl.remove_edges(bg, [1, 2], etype="plays")
    assert bg.batch_size == bg_r.batch_size
    for nty in ntypes:
        assert F.array_equal(bg.batch_num_nodes(nty), bg_r.batch_num_nodes(nty))
    assert F.array_equal(
        bg.batch_num_edges("follows"), bg_r.batch_num_edges("follows")
    )
    assert F.array_equal(
        bg_r.batch_num_edges("plays"), F.tensor([1, 0, 1], dtype=idtype)
    )

    bg_r = dgl.remove_edges(
        bg, F.tensor([0, 1, 3], dtype=idtype), etype="follows"
    )
    assert bg.batch_size == bg_r.batch_size
    for nty in ntypes:
        assert F.array_equal(bg.batch_num_nodes(nty), bg_r.batch_num_nodes(nty))
    assert F.array_equal(
        bg_r.batch_num_edges("follows"), F.tensor([0, 1, 0], dtype=idtype)
    )
    assert F.array_equal(
        bg.batch_num_edges("plays"), bg_r.batch_num_edges("plays")
    )

    bg_r = dgl.remove_edges(bg, F.tensor([1, 2], dtype=idtype), etype="plays")
    assert bg.batch_size == bg_r.batch_size
    for nty in ntypes:
        assert F.array_equal(bg.batch_num_nodes(nty), bg_r.batch_num_nodes(nty))
    assert F.array_equal(
        bg.batch_num_edges("follows"), bg_r.batch_num_edges("follows")
    )
    assert F.array_equal(
        bg_r.batch_num_edges("plays"), F.tensor([1, 0, 1], dtype=idtype)
    )


@parametrize_idtype
def test_remove_nodes(idtype):
    # homogeneous Graphs
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    n = 0
    g = dgl.remove_nodes(g, n)
    assert g.num_nodes() == 2
    assert g.num_edges() == 1
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([0], dtype=idtype))
    assert F.array_equal(v, F.tensor([1], dtype=idtype))
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    n = [1]
    g = dgl.remove_nodes(g, n)
    assert g.num_nodes() == 2
    assert g.num_edges() == 0
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    n = F.tensor([2], dtype=idtype)
    g = dgl.remove_nodes(g, n)
    assert g.num_nodes() == 2
    assert g.num_edges() == 1
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([0], dtype=idtype))
    assert F.array_equal(v, F.tensor([1], dtype=idtype))

    # invalid nid
    assert_fail = False
    try:
        g.remove_nodes(3)
    except:
        assert_fail = True
    assert assert_fail

    # has node and edge data
    g = dgl.graph(([0, 0, 2], [0, 1, 2]), idtype=idtype, device=F.ctx())
    g.ndata["hv"] = F.copy_to(F.tensor([1, 2, 3], dtype=idtype), ctx=F.ctx())
    g.edata["he"] = F.copy_to(F.tensor([1, 2, 3], dtype=idtype), ctx=F.ctx())
    g = dgl.remove_nodes(g, F.tensor([0], dtype=idtype))
    assert g.num_nodes() == 2
    assert g.num_edges() == 1
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([1], dtype=idtype))
    assert F.array_equal(v, F.tensor([1], dtype=idtype))
    assert F.array_equal(g.ndata["hv"], F.tensor([2, 3], dtype=idtype))
    assert F.array_equal(g.edata["he"], F.tensor([3], dtype=idtype))

    # node id larger than current max node id
    g = dgl.heterograph(
        {("user", "plays", "game"): ([0, 1], [1, 2])},
        idtype=idtype,
        device=F.ctx(),
    )
    n = 0
    g = dgl.remove_nodes(g, n, ntype="user")
    assert g.num_nodes("user") == 1
    assert g.num_nodes("game") == 3
    assert g.num_edges() == 1
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([0], dtype=idtype))
    assert F.array_equal(v, F.tensor([2], dtype=idtype))
    g = dgl.heterograph(
        {("user", "plays", "game"): ([0, 1], [1, 2])},
        idtype=idtype,
        device=F.ctx(),
    )
    n = [1]
    g = dgl.remove_nodes(g, n, ntype="user")
    assert g.num_nodes("user") == 1
    assert g.num_nodes("game") == 3
    assert g.num_edges() == 1
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([0], dtype=idtype))
    assert F.array_equal(v, F.tensor([1], dtype=idtype))
    g = dgl.heterograph(
        {("user", "plays", "game"): ([0, 1], [1, 2])},
        idtype=idtype,
        device=F.ctx(),
    )
    n = F.tensor([0], dtype=idtype)
    g = dgl.remove_nodes(g, n, ntype="game")
    assert g.num_nodes("user") == 2
    assert g.num_nodes("game") == 2
    assert g.num_edges() == 2
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([0, 1], dtype=idtype))
    assert F.array_equal(v, F.tensor([0, 1], dtype=idtype))

    # heterogeneous graph
    g = create_test_heterograph3(idtype)
    g.edges["plays"].data["h"] = F.copy_to(
        F.tensor([1, 2, 3, 4], dtype=idtype), ctx=F.ctx()
    )
    g = dgl.remove_nodes(g, 0, ntype="game")
    assert g.num_nodes("user") == 3
    assert g.num_nodes("game") == 1
    assert g.num_nodes("developer") == 2
    assert g.num_edges("plays") == 2
    assert g.num_edges("develops") == 1
    assert F.array_equal(
        g.nodes["user"].data["h"], F.tensor([1, 1, 1], dtype=idtype)
    )
    assert F.array_equal(g.nodes["game"].data["h"], F.tensor([2], dtype=idtype))
    assert F.array_equal(
        g.nodes["developer"].data["h"], F.tensor([3, 3], dtype=idtype)
    )
    u, v = g.edges(form="uv", order="eid", etype="plays")
    assert F.array_equal(u, F.tensor([1, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([0, 0], dtype=idtype))
    assert F.array_equal(
        g.edges["plays"].data["h"], F.tensor([3, 4], dtype=idtype)
    )
    u, v = g.edges(form="uv", order="eid", etype="develops")
    assert F.array_equal(u, F.tensor([1], dtype=idtype))
    assert F.array_equal(v, F.tensor([0], dtype=idtype))

    # batched graph
    ctx = F.ctx()
    g1 = dgl.graph(([0, 1], [1, 2]), num_nodes=5, idtype=idtype, device=ctx)
    g2 = dgl.graph(([], []), idtype=idtype, device=ctx)
    g3 = dgl.graph(([2, 3, 4], [3, 2, 1]), idtype=idtype, device=ctx)
    bg = dgl.batch([g1, g2, g3])
    bg_r = dgl.remove_nodes(bg, 1)
    assert bg_r.batch_size == bg.batch_size
    assert F.array_equal(
        bg_r.batch_num_nodes(), F.tensor([4, 0, 5], dtype=idtype)
    )
    assert F.array_equal(
        bg_r.batch_num_edges(), F.tensor([0, 0, 3], dtype=idtype)
    )

    bg_r = dgl.remove_nodes(bg, [1, 7])
    assert bg_r.batch_size == bg.batch_size
    assert F.array_equal(
        bg_r.batch_num_nodes(), F.tensor([4, 0, 4], dtype=idtype)
    )
    assert F.array_equal(
        bg_r.batch_num_edges(), F.tensor([0, 0, 1], dtype=idtype)
    )

    bg_r = dgl.remove_nodes(bg, F.tensor([1, 7], dtype=idtype))
    assert bg_r.batch_size == bg.batch_size
    assert F.array_equal(
        bg_r.batch_num_nodes(), F.tensor([4, 0, 4], dtype=idtype)
    )
    assert F.array_equal(
        bg_r.batch_num_edges(), F.tensor([0, 0, 1], dtype=idtype)
    )

    # batched heterogeneous graph
    g1 = dgl.heterograph(
        {
            ("user", "follows", "user"): ([0, 1], [1, 2]),
            ("user", "plays", "game"): ([1, 3], [0, 1]),
        },
        num_nodes_dict={"user": 4, "game": 3},
        idtype=idtype,
        device=ctx,
    )
    g2 = dgl.heterograph(
        {
            ("user", "follows", "user"): ([0, 2], [3, 4]),
            ("user", "plays", "game"): ([], []),
        },
        num_nodes_dict={"user": 6, "game": 2},
        idtype=idtype,
        device=ctx,
    )
    g3 = dgl.heterograph(
        {
            ("user", "follows", "user"): ([], []),
            ("user", "plays", "game"): ([1, 2], [1, 2]),
        },
        idtype=idtype,
        device=ctx,
    )
    bg = dgl.batch([g1, g2, g3])
    bg_r = dgl.remove_nodes(bg, 1, ntype="user")
    assert bg_r.batch_size == bg.batch_size
    assert F.array_equal(
        bg_r.batch_num_nodes("user"), F.tensor([3, 6, 3], dtype=idtype)
    )
    assert F.array_equal(
        bg.batch_num_nodes("game"), bg_r.batch_num_nodes("game")
    )
    assert F.array_equal(
        bg_r.batch_num_edges("follows"), F.tensor([0, 2, 0], dtype=idtype)
    )
    assert F.array_equal(
        bg_r.batch_num_edges("plays"), F.tensor([1, 0, 2], dtype=idtype)
    )

    bg_r = dgl.remove_nodes(bg, 6, ntype="game")
    assert bg_r.batch_size == bg.batch_size
    assert F.array_equal(
        bg.batch_num_nodes("user"), bg_r.batch_num_nodes("user")
    )
    assert F.array_equal(
        bg_r.batch_num_nodes("game"), F.tensor([3, 2, 2], dtype=idtype)
    )
    assert F.array_equal(
        bg.batch_num_edges("follows"), bg_r.batch_num_edges("follows")
    )
    assert F.array_equal(
        bg_r.batch_num_edges("plays"), F.tensor([2, 0, 1], dtype=idtype)
    )

    bg_r = dgl.remove_nodes(bg, [1, 5, 6, 11], ntype="user")
    assert bg_r.batch_size == bg.batch_size
    assert F.array_equal(
        bg_r.batch_num_nodes("user"), F.tensor([3, 4, 2], dtype=idtype)
    )
    assert F.array_equal(
        bg.batch_num_nodes("game"), bg_r.batch_num_nodes("game")
    )
    assert F.array_equal(
        bg_r.batch_num_edges("follows"), F.tensor([0, 1, 0], dtype=idtype)
    )
    assert F.array_equal(
        bg_r.batch_num_edges("plays"), F.tensor([1, 0, 1], dtype=idtype)
    )

    bg_r = dgl.remove_nodes(bg, [0, 3, 4, 7], ntype="game")
    assert bg_r.batch_size == bg.batch_size
    assert F.array_equal(
        bg.batch_num_nodes("user"), bg_r.batch_num_nodes("user")
    )
    assert F.array_equal(
        bg_r.batch_num_nodes("game"), F.tensor([2, 0, 2], dtype=idtype)
    )
    assert F.array_equal(
        bg.batch_num_edges("follows"), bg_r.batch_num_edges("follows")
    )
    assert F.array_equal(
        bg_r.batch_num_edges("plays"), F.tensor([1, 0, 1], dtype=idtype)
    )

    bg_r = dgl.remove_nodes(
        bg, F.tensor([1, 5, 6, 11], dtype=idtype), ntype="user"
    )
    assert bg_r.batch_size == bg.batch_size
    assert F.array_equal(
        bg_r.batch_num_nodes("user"), F.tensor([3, 4, 2], dtype=idtype)
    )
    assert F.array_equal(
        bg.batch_num_nodes("game"), bg_r.batch_num_nodes("game")
    )
    assert F.array_equal(
        bg_r.batch_num_edges("follows"), F.tensor([0, 1, 0], dtype=idtype)
    )
    assert F.array_equal(
        bg_r.batch_num_edges("plays"), F.tensor([1, 0, 1], dtype=idtype)
    )

    bg_r = dgl.remove_nodes(
        bg, F.tensor([0, 3, 4, 7], dtype=idtype), ntype="game"
    )
    assert bg_r.batch_size == bg.batch_size
    assert F.array_equal(
        bg.batch_num_nodes("user"), bg_r.batch_num_nodes("user")
    )
    assert F.array_equal(
        bg_r.batch_num_nodes("game"), F.tensor([2, 0, 2], dtype=idtype)
    )
    assert F.array_equal(
        bg.batch_num_edges("follows"), bg_r.batch_num_edges("follows")
    )
    assert F.array_equal(
        bg_r.batch_num_edges("plays"), F.tensor([1, 0, 1], dtype=idtype)
    )


@parametrize_idtype
def test_add_selfloop(idtype):
    # homogeneous graph

    # test for fill_data is float
    g = dgl.graph(([0, 0, 2], [2, 1, 0]), idtype=idtype, device=F.ctx())
    g.edata["he"] = F.copy_to(F.tensor([1, 2, 3], dtype=idtype), ctx=F.ctx())
    g.edata["he1"] = F.copy_to(
        F.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]), ctx=F.ctx()
    )
    g.ndata["hn"] = F.copy_to(F.tensor([1, 2, 3], dtype=idtype), ctx=F.ctx())
    g = dgl.add_self_loop(g)
    assert g.num_nodes() == 3
    assert g.num_edges() == 6
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([0, 0, 2, 0, 1, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([2, 1, 0, 0, 1, 2], dtype=idtype))
    assert F.array_equal(
        g.edata["he"], F.tensor([1, 2, 3, 1, 1, 1], dtype=idtype)
    )
    assert F.array_equal(
        g.edata["he1"],
        F.tensor(
            [
                [0.0, 1.0],
                [2.0, 3.0],
                [4.0, 5.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        ),
    )

    # test for fill_data is int
    g = dgl.graph(([0, 0, 2], [2, 1, 0]), idtype=idtype, device=F.ctx())
    g.edata["he"] = F.copy_to(F.tensor([1, 2, 3], dtype=idtype), ctx=F.ctx())
    g.edata["he1"] = F.copy_to(
        F.tensor([[0, 1], [2, 3], [4, 5]], dtype=idtype), ctx=F.ctx()
    )
    g.ndata["hn"] = F.copy_to(F.tensor([1, 2, 3], dtype=idtype), ctx=F.ctx())
    g = dgl.add_self_loop(g, fill_data=1)
    assert g.num_nodes() == 3
    assert g.num_edges() == 6
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([0, 0, 2, 0, 1, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([2, 1, 0, 0, 1, 2], dtype=idtype))
    assert F.array_equal(
        g.edata["he"], F.tensor([1, 2, 3, 1, 1, 1], dtype=idtype)
    )
    assert F.array_equal(
        g.edata["he1"],
        F.tensor(
            [[0, 1], [2, 3], [4, 5], [1, 1], [1, 1], [1, 1]], dtype=idtype
        ),
    )

    # test for fill_data is str
    g = dgl.graph(([0, 0, 2], [2, 1, 0]), idtype=idtype, device=F.ctx())
    g.edata["he"] = F.copy_to(F.tensor([1.0, 2.0, 3.0]), ctx=F.ctx())
    g.edata["he1"] = F.copy_to(
        F.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]), ctx=F.ctx()
    )
    g.ndata["hn"] = F.copy_to(F.tensor([1, 2, 3], dtype=idtype), ctx=F.ctx())
    g = dgl.add_self_loop(g, fill_data="sum")
    assert g.num_nodes() == 3
    assert g.num_edges() == 6
    u, v = g.edges(form="uv", order="eid")
    assert F.array_equal(u, F.tensor([0, 0, 2, 0, 1, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([2, 1, 0, 0, 1, 2], dtype=idtype))
    assert F.array_equal(
        g.edata["he"], F.tensor([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])
    )
    assert F.array_equal(
        g.edata["he1"],
        F.tensor(
            [
                [0.0, 1.0],
                [2.0, 3.0],
                [4.0, 5.0],
                [4.0, 5.0],
                [2.0, 3.0],
                [0.0, 1.0],
            ]
        ),
    )

    # bipartite graph
    g = dgl.heterograph(
        {("user", "plays", "game"): ([0, 1, 2], [1, 2, 2])},
        idtype=idtype,
        device=F.ctx(),
    )
    # nothing will happend
    raise_error = False
    try:
        g = dgl.add_self_loop(g)
    except:
        raise_error = True
    assert raise_error

    # test for fill_data is float
    g = create_test_heterograph5(idtype)
    g.edges["follows"].data["h1"] = F.copy_to(
        F.tensor([[0.0, 1.0], [1.0, 2.0]]), ctx=F.ctx()
    )
    g = dgl.add_self_loop(g, etype="follows")
    assert g.num_nodes("user") == 3
    assert g.num_nodes("game") == 2
    assert g.num_edges("follows") == 5
    assert g.num_edges("plays") == 2
    u, v = g.edges(form="uv", order="eid", etype="follows")
    assert F.array_equal(u, F.tensor([1, 2, 0, 1, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([0, 1, 0, 1, 2], dtype=idtype))
    assert F.array_equal(
        g.edges["follows"].data["h"], F.tensor([1, 2, 1, 1, 1], dtype=idtype)
    )
    assert F.array_equal(
        g.edges["follows"].data["h1"],
        F.tensor([[0.0, 1.0], [1.0, 2.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
    )
    assert F.array_equal(
        g.edges["plays"].data["h"], F.tensor([1, 2], dtype=idtype)
    )

    # test for fill_data is int
    g = create_test_heterograph5(idtype)
    g.edges["follows"].data["h1"] = F.copy_to(
        F.tensor([[0, 1], [1, 2]], dtype=idtype), ctx=F.ctx()
    )
    g = dgl.add_self_loop(g, fill_data=1, etype="follows")
    assert g.num_nodes("user") == 3
    assert g.num_nodes("game") == 2
    assert g.num_edges("follows") == 5
    assert g.num_edges("plays") == 2
    u, v = g.edges(form="uv", order="eid", etype="follows")
    assert F.array_equal(u, F.tensor([1, 2, 0, 1, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([0, 1, 0, 1, 2], dtype=idtype))
    assert F.array_equal(
        g.edges["follows"].data["h"], F.tensor([1, 2, 1, 1, 1], dtype=idtype)
    )
    assert F.array_equal(
        g.edges["follows"].data["h1"],
        F.tensor([[0, 1], [1, 2], [1, 1], [1, 1], [1, 1]], dtype=idtype),
    )
    assert F.array_equal(
        g.edges["plays"].data["h"], F.tensor([1, 2], dtype=idtype)
    )

    # test for fill_data is str
    g = dgl.heterograph(
        {
            ("user", "follows", "user"): (
                F.tensor([1, 2], dtype=idtype),
                F.tensor([0, 1], dtype=idtype),
            ),
            ("user", "plays", "game"): (
                F.tensor([0, 1], dtype=idtype),
                F.tensor([0, 1], dtype=idtype),
            ),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    g.nodes["user"].data["h"] = F.copy_to(
        F.tensor([1, 1, 1], dtype=idtype), ctx=F.ctx()
    )
    g.nodes["game"].data["h"] = F.copy_to(
        F.tensor([2, 2], dtype=idtype), ctx=F.ctx()
    )
    g.edges["follows"].data["h"] = F.copy_to(F.tensor([1.0, 2.0]), ctx=F.ctx())
    g.edges["follows"].data["h1"] = F.copy_to(
        F.tensor([[0.0, 1.0], [1.0, 2.0]]), ctx=F.ctx()
    )
    g.edges["plays"].data["h"] = F.copy_to(F.tensor([1.0, 2.0]), ctx=F.ctx())
    g = dgl.add_self_loop(g, fill_data="mean", etype="follows")
    assert g.num_nodes("user") == 3
    assert g.num_nodes("game") == 2
    assert g.num_edges("follows") == 5
    assert g.num_edges("plays") == 2
    u, v = g.edges(form="uv", order="eid", etype="follows")
    assert F.array_equal(u, F.tensor([1, 2, 0, 1, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([0, 1, 0, 1, 2], dtype=idtype))
    assert F.array_equal(
        g.edges["follows"].data["h"], F.tensor([1.0, 2.0, 1.0, 2.0, 0.0])
    )
    assert F.array_equal(
        g.edges["follows"].data["h1"],
        F.tensor([[0.0, 1.0], [1.0, 2.0], [0.0, 1.0], [1.0, 2.0], [0.0, 0.0]]),
    )
    assert F.array_equal(g.edges["plays"].data["h"], F.tensor([1.0, 2.0]))

    raise_error = False
    try:
        g = dgl.add_self_loop(g, etype="plays")
    except:
        raise_error = True
    assert raise_error


@parametrize_idtype
def test_remove_selfloop(idtype):
    # homogeneous graph
    g = dgl.graph(([0, 0, 0, 1], [1, 0, 0, 2]), idtype=idtype, device=F.ctx())
    g.edata["he"] = F.copy_to(F.tensor([1, 2, 3, 4], dtype=idtype), ctx=F.ctx())
    g = dgl.remove_self_loop(g)
    assert g.num_nodes() == 3
    assert g.num_edges() == 2
    assert F.array_equal(g.edata["he"], F.tensor([1, 4], dtype=idtype))

    # bipartite graph
    g = dgl.heterograph(
        {("user", "plays", "game"): ([0, 1, 2], [1, 2, 2])},
        idtype=idtype,
        device=F.ctx(),
    )
    # nothing will happend
    raise_error = False
    try:
        g = dgl.remove_self_loop(g, etype="plays")
    except:
        raise_error = True
    assert raise_error

    g = create_test_heterograph4(idtype)
    g = dgl.remove_self_loop(g, etype="follows")
    assert g.num_nodes("user") == 3
    assert g.num_nodes("game") == 2
    assert g.num_edges("follows") == 2
    assert g.num_edges("plays") == 2
    u, v = g.edges(form="uv", order="eid", etype="follows")
    assert F.array_equal(u, F.tensor([1, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([0, 1], dtype=idtype))
    assert F.array_equal(
        g.edges["follows"].data["h"], F.tensor([2, 4], dtype=idtype)
    )
    assert F.array_equal(
        g.edges["plays"].data["h"], F.tensor([1, 2], dtype=idtype)
    )

    raise_error = False
    try:
        g = dgl.remove_self_loop(g, etype="plays")
    except:
        raise_error = True
    assert raise_error

    # batch information
    g = dgl.graph(
        ([0, 0, 0, 1, 3, 3, 4], [1, 0, 0, 2, 3, 4, 4]),
        idtype=idtype,
        device=F.ctx(),
    )
    g.set_batch_num_nodes([3, 2])
    g.set_batch_num_edges([4, 3])
    g = dgl.remove_self_loop(g)
    assert g.num_nodes() == 5
    assert g.num_edges() == 3
    assert F.array_equal(g.batch_num_nodes(), F.tensor([3, 2], dtype=idtype))
    assert F.array_equal(g.batch_num_edges(), F.tensor([2, 1], dtype=idtype))


@parametrize_idtype
def test_reorder_graph(idtype):
    g = dgl.graph(
        ([0, 1, 2, 3, 4], [2, 2, 3, 2, 3]), idtype=idtype, device=F.ctx()
    )
    g.ndata["h"] = F.copy_to(F.randn((g.num_nodes(), 3)), ctx=F.ctx())
    g.edata["w"] = F.copy_to(F.randn((g.num_edges(), 2)), ctx=F.ctx())

    # call with default: node_permute_algo=None, edge_permute_algo='src'
    rg = dgl.reorder_graph(g)
    assert dgl.EID in rg.edata.keys()
    src = F.asnumpy(rg.edges()[0])
    assert np.array_equal(src, np.sort(src))

    # call with 'rcmk' node_permute_algo
    rg = dgl.reorder_graph(g, node_permute_algo="rcmk")
    assert dgl.NID in rg.ndata.keys()
    assert dgl.EID in rg.edata.keys()
    src = F.asnumpy(rg.edges()[0])
    assert np.array_equal(src, np.sort(src))

    # call with 'dst' edge_permute_algo
    rg = dgl.reorder_graph(g, edge_permute_algo="dst")
    dst = F.asnumpy(rg.edges()[1])
    assert np.array_equal(dst, np.sort(dst))

    # call with unknown edge_permute_algo
    raise_error = False
    try:
        dgl.reorder_graph(g, edge_permute_algo="none")
    except:
        raise_error = True
    assert raise_error

    # reorder back to original according to stored ids
    rg = dgl.reorder_graph(g, node_permute_algo="rcmk")
    rg2 = dgl.reorder_graph(
        rg,
        "custom",
        permute_config={"nodes_perm": np.argsort(F.asnumpy(rg.ndata[dgl.NID]))},
    )
    assert F.array_equal(g.ndata["h"], rg2.ndata["h"])
    assert F.array_equal(g.edata["w"], rg2.edata["w"])

    # do not store ids
    rg = dgl.reorder_graph(g, store_ids=False)
    assert not dgl.NID in rg.ndata.keys()
    assert not dgl.EID in rg.edata.keys()

    # metis does not work on windows.
    if os.name == "nt":
        pass
    else:
        # metis_partition may fail for small graph.
        mg = create_large_graph(1000).to(F.ctx())

        # call with metis strategy, but k is not specified
        raise_error = False
        try:
            dgl.reorder_graph(mg, node_permute_algo="metis")
        except:
            raise_error = True
        assert raise_error

        # call with metis strategy, k is specified
        raise_error = False
        try:
            dgl.reorder_graph(
                mg, node_permute_algo="metis", permute_config={"k": 2}
            )
        except:
            raise_error = True
        assert not raise_error

    # call with qualified nodes_perm specified
    nodes_perm = np.random.permutation(g.num_nodes())
    raise_error = False
    try:
        dgl.reorder_graph(
            g,
            node_permute_algo="custom",
            permute_config={"nodes_perm": nodes_perm},
        )
    except:
        raise_error = True
    assert not raise_error

    # call with unqualified nodes_perm specified
    raise_error = False
    try:
        dgl.reorder_graph(
            g,
            node_permute_algo="custom",
            permute_config={"nodes_perm": nodes_perm[: g.num_nodes() - 1]},
        )
    except:
        raise_error = True
    assert raise_error

    # call with unsupported strategy
    raise_error = False
    try:
        dgl.reorder_graph(g, node_permute_algo="cmk")
    except:
        raise_error = True
    assert raise_error

    # heterograph: not supported
    raise_error = False
    try:
        hg = dgl.heterogrpah(
            {("user", "follow", "user"): ([0, 1], [1, 2])},
            idtype=idtype,
            device=F.ctx(),
        )
        dgl.reorder_graph(hg)
    except:
        raise_error = True
    assert raise_error

    # TODO: shall we fix them?
    # add 'csc' format if needed
    # fg = g.formats('csr')
    # assert 'csc' not in sum(fg.formats().values(), [])
    # rfg = dgl.reorder_graph(fg)
    # assert 'csc' in sum(rfg.formats().values(), [])


@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow",
    reason="TF doesn't support a slicing operation",
)
@parametrize_idtype
def test_norm_by_dst(idtype):
    # Case1: A homogeneous graph
    g = dgl.graph(([0, 1, 1], [1, 1, 2]), idtype=idtype, device=F.ctx())
    eweight = dgl.norm_by_dst(g)
    assert F.allclose(eweight, F.tensor([0.5, 0.5, 1.0]))

    # Case2: A heterogeneous graph
    g = dgl.heterograph(
        {
            ("user", "follows", "user"): ([0, 1], [1, 2]),
            ("user", "plays", "game"): ([0, 1, 1], [1, 1, 2]),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    eweight = dgl.norm_by_dst(g, etype=("user", "plays", "game"))
    assert F.allclose(eweight, F.tensor([0.5, 0.5, 1.0]))


@parametrize_idtype
def test_module_add_self_loop(idtype):
    g = dgl.graph(([1, 1], [1, 2]), idtype=idtype, device=F.ctx())
    g.ndata["h"] = F.randn((g.num_nodes(), 2))
    g.edata["w"] = F.randn((g.num_edges(), 3))

    # Case1: add self-loops with the default setting
    transform = dgl.AddSelfLoop()
    new_g = transform(g)
    assert new_g.device == g.device
    assert new_g.idtype == g.idtype
    assert new_g.num_nodes() == g.num_nodes()
    assert new_g.num_edges() == 4
    src, dst = new_g.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 0), (1, 1), (1, 2), (2, 2)}
    assert "h" in new_g.ndata
    assert "w" in new_g.edata

    # Case2: remove self-loops first to avoid duplicate ones
    transform = dgl.AddSelfLoop(allow_duplicate=True)
    new_g = transform(g)
    assert new_g.device == g.device
    assert new_g.idtype == g.idtype
    assert new_g.num_nodes() == g.num_nodes()
    assert new_g.num_edges() == 5
    src, dst = new_g.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 0), (1, 1), (1, 2), (2, 2)}
    assert "h" in new_g.ndata
    assert "w" in new_g.edata

    # Case3: add self-loops for a homogeneous graph (the example in doc)
    transform = dgl.AddSelfLoop(fill_data="sum")
    g = dgl.graph(([0, 0, 2], [2, 1, 0]), idtype=idtype, device=F.ctx())
    new_g = transform(g)
    assert new_g.device == g.device
    assert new_g.idtype == g.idtype
    assert new_g.num_nodes() == g.num_nodes()
    src, dst = new_g.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 2), (0, 1), (2, 0), (0, 0), (1, 1), (2, 2)}

    # Create a heterogeneous graph
    g = dgl.heterograph(
        {
            ("user", "plays", "game"): ([0], [1]),
            ("user", "follows", "user"): ([1], [3]),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    g.nodes["user"].data["h1"] = F.randn((4, 2))
    g.edges["plays"].data["w1"] = F.randn((1, 3))
    g.nodes["game"].data["h2"] = F.randn((2, 4))
    g.edges["follows"].data["w2"] = F.randn((1, 5))

    # Case4: add self-loops for a heterogeneous graph
    new_g = transform(g)
    assert new_g.device == g.device
    assert new_g.idtype == g.idtype
    assert new_g.ntypes == g.ntypes
    assert new_g.canonical_etypes == g.canonical_etypes
    for nty in new_g.ntypes:
        assert new_g.num_nodes(nty) == g.num_nodes(nty)
    assert new_g.num_edges("plays") == 1
    assert new_g.num_edges("follows") == 5
    assert "h1" in new_g.nodes["user"].data
    assert "h2" in new_g.nodes["game"].data
    assert "w1" in new_g.edges["plays"].data
    assert "w2" in new_g.edges["follows"].data

    # Case5: add self-etypes for a heterogeneous graph
    transform = dgl.AddSelfLoop(new_etypes=True)
    new_g = transform(g)
    assert new_g.device == g.device
    assert new_g.idtype == g.idtype
    assert new_g.ntypes == g.ntypes
    assert set(new_g.canonical_etypes) == {
        ("user", "plays", "game"),
        ("user", "follows", "user"),
        ("user", "self", "user"),
        ("game", "self", "game"),
    }
    for nty in new_g.ntypes:
        assert new_g.num_nodes(nty) == g.num_nodes(nty)
    assert new_g.num_edges("plays") == 1
    assert new_g.num_edges("follows") == 5
    assert new_g.num_edges(("user", "self", "user")) == 4
    assert new_g.num_edges(("game", "self", "game")) == 2
    assert "h1" in new_g.nodes["user"].data
    assert "h2" in new_g.nodes["game"].data
    assert "w1" in new_g.edges["plays"].data
    assert "w2" in new_g.edges["follows"].data


@parametrize_idtype
def test_module_remove_self_loop(idtype):
    transform = dgl.RemoveSelfLoop()

    # Case1: homogeneous graph
    g = dgl.graph(([1, 1], [1, 2]), idtype=idtype, device=F.ctx())
    g.ndata["h"] = F.randn((g.num_nodes(), 2))
    g.edata["w"] = F.randn((g.num_edges(), 3))
    new_g = transform(g)
    assert new_g.device == g.device
    assert new_g.idtype == g.idtype
    assert new_g.num_nodes() == g.num_nodes()
    assert new_g.num_edges() == 1
    src, dst = new_g.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(1, 2)}
    assert "h" in new_g.ndata
    assert "w" in new_g.edata

    # Case2: heterogeneous graph
    g = dgl.heterograph(
        {
            ("user", "plays", "game"): ([0, 1], [1, 1]),
            ("user", "follows", "user"): ([1, 2], [2, 2]),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    g.nodes["user"].data["h1"] = F.randn((3, 2))
    g.edges["plays"].data["w1"] = F.randn((2, 3))
    g.nodes["game"].data["h2"] = F.randn((2, 4))
    g.edges["follows"].data["w2"] = F.randn((2, 5))

    new_g = transform(g)
    assert new_g.device == g.device
    assert new_g.idtype == g.idtype
    assert new_g.ntypes == g.ntypes
    assert new_g.canonical_etypes == g.canonical_etypes
    for nty in new_g.ntypes:
        assert new_g.num_nodes(nty) == g.num_nodes(nty)
    assert new_g.num_edges("plays") == 2
    assert new_g.num_edges("follows") == 1
    assert "h1" in new_g.nodes["user"].data
    assert "h2" in new_g.nodes["game"].data
    assert "w1" in new_g.edges["plays"].data
    assert "w2" in new_g.edges["follows"].data


@parametrize_idtype
def test_module_add_reverse(idtype):
    transform = dgl.AddReverse()

    # Case1: Add reverse edges for a homogeneous graph
    g = dgl.graph(([0], [1]), idtype=idtype, device=F.ctx())
    g.ndata["h"] = F.randn((g.num_nodes(), 3))
    g.edata["w"] = F.randn((g.num_edges(), 2))
    new_g = transform(g)
    assert new_g.device == g.device
    assert new_g.idtype == g.idtype
    assert g.num_nodes() == new_g.num_nodes()
    src, dst = new_g.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 1), (1, 0)}
    assert F.allclose(g.ndata["h"], new_g.ndata["h"])
    assert F.allclose(g.edata["w"], F.narrow_row(new_g.edata["w"], 0, 1))
    assert F.allclose(
        F.narrow_row(new_g.edata["w"], 1, 2),
        F.zeros((1, 2), F.float32, F.ctx()),
    )

    # Case2: Add reverse edges for a homogeneous graph and copy edata
    transform = dgl.AddReverse(copy_edata=True)
    new_g = transform(g)
    assert new_g.device == g.device
    assert new_g.idtype == g.idtype
    assert g.num_nodes() == new_g.num_nodes()
    src, dst = new_g.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 1), (1, 0)}
    assert F.allclose(g.ndata["h"], new_g.ndata["h"])
    assert F.allclose(g.edata["w"], F.narrow_row(new_g.edata["w"], 0, 1))
    assert F.allclose(g.edata["w"], F.narrow_row(new_g.edata["w"], 1, 2))

    # Case3: Add reverse edges for a heterogeneous graph
    g = dgl.heterograph(
        {
            ("user", "plays", "game"): ([0, 1], [1, 1]),
            ("user", "follows", "user"): ([1, 2], [2, 2]),
        },
        device=F.ctx(),
    )
    new_g = transform(g)
    assert new_g.device == g.device
    assert new_g.idtype == g.idtype
    assert g.ntypes == new_g.ntypes
    assert set(new_g.canonical_etypes) == {
        ("user", "plays", "game"),
        ("user", "follows", "user"),
        ("game", "rev_plays", "user"),
    }
    for nty in g.ntypes:
        assert g.num_nodes(nty) == new_g.num_nodes(nty)

    src, dst = new_g.edges(etype="plays")
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 1), (1, 1)}

    src, dst = new_g.edges(etype="follows")
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(1, 2), (2, 2), (2, 1)}

    src, dst = new_g.edges(etype="rev_plays")
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(1, 1), (1, 0)}

    # Case4: Enforce reverse edge types for symmetric canonical edge types
    transform = dgl.AddReverse(sym_new_etype=True)
    new_g = transform(g)
    assert new_g.device == g.device
    assert new_g.idtype == g.idtype
    assert g.ntypes == new_g.ntypes
    assert set(new_g.canonical_etypes) == {
        ("user", "plays", "game"),
        ("user", "follows", "user"),
        ("game", "rev_plays", "user"),
        ("user", "rev_follows", "user"),
    }
    for nty in g.ntypes:
        assert g.num_nodes(nty) == new_g.num_nodes(nty)

    src, dst = new_g.edges(etype="plays")
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 1), (1, 1)}

    src, dst = new_g.edges(etype="follows")
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(1, 2), (2, 2)}

    src, dst = new_g.edges(etype="rev_plays")
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(1, 1), (1, 0)}

    src, dst = new_g.edges(etype="rev_follows")
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(2, 1), (2, 2)}


@unittest.skipIf(
    F._default_context_str == "gpu", reason="GPU not supported for to_simple"
)
@parametrize_idtype
def test_module_to_simple(idtype):
    transform = dgl.ToSimple()
    g = dgl.graph(([0, 1, 1], [1, 2, 2]), idtype=idtype, device=F.ctx())
    g.ndata["h"] = F.randn((g.num_nodes(), 2))
    g.edata["w"] = F.tensor([[0.1], [0.2], [0.3]])
    sg = transform(g)
    assert sg.device == g.device
    assert sg.idtype == g.idtype
    assert sg.num_nodes() == g.num_nodes()
    assert sg.num_edges() == 2
    src, dst = sg.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 1), (1, 2)}
    assert F.allclose(sg.edata["count"], F.tensor([1, 2]))
    assert F.allclose(sg.ndata["h"], g.ndata["h"])

    g = dgl.heterograph(
        {
            ("user", "follows", "user"): ([0, 1, 1], [1, 2, 2]),
            ("user", "plays", "game"): ([0, 1, 0], [1, 1, 1]),
        }
    )
    sg = transform(g)
    assert sg.device == g.device
    assert sg.idtype == g.idtype
    assert sg.ntypes == g.ntypes
    assert sg.canonical_etypes == g.canonical_etypes
    for nty in sg.ntypes:
        assert sg.num_nodes(nty) == g.num_nodes(nty)
    for ety in sg.canonical_etypes:
        assert sg.num_edges(ety) == 2

    src, dst = sg.edges(etype="follows")
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 1), (1, 2)}

    src, dst = sg.edges(etype="plays")
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 1), (1, 1)}


@parametrize_idtype
def test_module_line_graph(idtype):
    transform = dgl.LineGraph()
    g = dgl.graph(([0, 1, 1], [1, 0, 2]), idtype=idtype, device=F.ctx())
    g.ndata["h"] = F.tensor([[0.0], [1.0], [2.0]])
    g.edata["w"] = F.tensor([[0.0], [0.1], [0.2]])
    new_g = transform(g)
    assert new_g.device == g.device
    assert new_g.idtype == g.idtype
    assert new_g.num_nodes() == g.num_edges()
    src, dst = new_g.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 1), (0, 2), (1, 0)}

    transform = dgl.LineGraph(backtracking=False)
    new_g = transform(g)
    assert new_g.device == g.device
    assert new_g.idtype == g.idtype
    assert new_g.num_nodes() == g.num_edges()
    src, dst = new_g.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 2)}


@parametrize_idtype
def test_module_khop_graph(idtype):
    transform = dgl.KHopGraph(2)
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    g.ndata["h"] = F.randn((g.num_nodes(), 2))
    new_g = transform(g)
    assert new_g.device == g.device
    assert new_g.idtype == g.idtype
    assert new_g.num_nodes() == g.num_nodes()
    assert F.allclose(g.ndata["h"], new_g.ndata["h"])
    src, dst = new_g.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 2)}


@parametrize_idtype
def test_module_add_metapaths(idtype):
    g = dgl.heterograph(
        {
            ("person", "author", "paper"): ([0, 0, 1], [1, 2, 2]),
            ("paper", "accepted", "venue"): ([1], [0]),
            ("paper", "rejected", "venue"): ([2], [1]),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    g.nodes["venue"].data["h"] = F.randn((g.num_nodes("venue"), 2))
    g.edges["author"].data["h"] = F.randn((g.num_edges("author"), 3))

    # Case1: keep_orig_edges is True
    metapaths = {
        "accepted": [
            ("person", "author", "paper"),
            ("paper", "accepted", "venue"),
        ],
        "rejected": [
            ("person", "author", "paper"),
            ("paper", "rejected", "venue"),
        ],
    }
    transform = dgl.AddMetaPaths(metapaths)
    new_g = transform(g)
    assert new_g.device == g.device
    assert new_g.idtype == g.idtype
    assert new_g.ntypes == g.ntypes
    assert set(new_g.canonical_etypes) == {
        ("person", "author", "paper"),
        ("paper", "accepted", "venue"),
        ("paper", "rejected", "venue"),
        ("person", "accepted", "venue"),
        ("person", "rejected", "venue"),
    }
    for nty in new_g.ntypes:
        assert new_g.num_nodes(nty) == g.num_nodes(nty)
    for ety in g.canonical_etypes:
        assert new_g.num_edges(ety) == g.num_edges(ety)
    assert F.allclose(
        g.nodes["venue"].data["h"], new_g.nodes["venue"].data["h"]
    )
    assert F.allclose(
        g.edges["author"].data["h"], new_g.edges["author"].data["h"]
    )

    src, dst = new_g.edges(etype=("person", "accepted", "venue"))
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 0)}

    src, dst = new_g.edges(etype=("person", "rejected", "venue"))
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 1), (1, 1)}

    # Case2: keep_orig_edges is False
    transform = dgl.AddMetaPaths(metapaths, keep_orig_edges=False)
    new_g = transform(g)
    assert new_g.device == g.device
    assert new_g.idtype == g.idtype
    assert new_g.ntypes == g.ntypes
    assert len(new_g.canonical_etypes) == 2
    for nty in new_g.ntypes:
        assert new_g.num_nodes(nty) == g.num_nodes(nty)
    assert F.allclose(
        g.nodes["venue"].data["h"], new_g.nodes["venue"].data["h"]
    )

    src, dst = new_g.edges(etype=("person", "accepted", "venue"))
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 0)}

    src, dst = new_g.edges(etype=("person", "rejected", "venue"))
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 1), (1, 1)}


@parametrize_idtype
def test_module_compose(idtype):
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    transform = dgl.Compose([dgl.AddReverse(), dgl.AddSelfLoop()])
    new_g = transform(g)
    assert new_g.device == g.device
    assert new_g.idtype == g.idtype
    assert new_g.num_edges() == 7

    src, dst = new_g.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 1), (1, 2), (1, 0), (2, 1), (0, 0), (1, 1), (2, 2)}


@parametrize_idtype
def test_module_gcnnorm(idtype):
    g = dgl.heterograph(
        {
            ("A", "r1", "A"): ([0, 1, 2], [0, 0, 1]),
            ("A", "r2", "B"): ([0, 0], [1, 1]),
            ("B", "r3", "B"): ([0, 1, 2], [0, 0, 1]),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    g.edges["r3"].data["w"] = F.tensor([0.1, 0.2, 0.3])
    transform = dgl.GCNNorm()
    new_g = transform(g)
    assert "w" not in new_g.edges[("A", "r2", "B")].data
    assert F.allclose(
        new_g.edges[("A", "r1", "A")].data["w"],
        F.tensor([1.0 / 2, 1.0 / math.sqrt(2), 0.0]),
    )
    assert F.allclose(
        new_g.edges[("B", "r3", "B")].data["w"],
        F.tensor([1.0 / 3, 2.0 / 3, 0.0]),
    )


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@parametrize_idtype
def test_module_ppr(idtype):
    g = dgl.graph(
        ([0, 1, 2, 3, 4], [2, 3, 4, 5, 3]), idtype=idtype, device=F.ctx()
    )
    g.ndata["h"] = F.randn((6, 2))
    transform = dgl.PPR(avg_degree=2)
    new_g = transform(g)
    assert new_g.idtype == g.idtype
    assert new_g.device == g.device
    assert new_g.num_nodes() == g.num_nodes()
    src, dst = new_g.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {
        (0, 0),
        (0, 2),
        (0, 4),
        (1, 1),
        (1, 3),
        (1, 5),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 3),
        (3, 5),
        (4, 3),
        (4, 4),
        (4, 5),
        (5, 5),
    }
    assert F.allclose(g.ndata["h"], new_g.ndata["h"])
    assert "w" in new_g.edata

    # Prior edge weights
    g.edata["w"] = F.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    new_g = transform(g)
    src, dst = new_g.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {
        (0, 0),
        (1, 1),
        (1, 3),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 3),
        (3, 5),
        (4, 3),
        (4, 4),
        (4, 5),
        (5, 5),
    }


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@parametrize_idtype
def test_module_heat_kernel(idtype):
    # Case1: directed graph
    g = dgl.graph(
        ([0, 1, 2, 3, 4], [2, 3, 4, 5, 3]), idtype=idtype, device=F.ctx()
    )
    g.ndata["h"] = F.randn((6, 2))
    transform = dgl.HeatKernel(avg_degree=1)
    new_g = transform(g)
    assert new_g.idtype == g.idtype
    assert new_g.device == g.device
    assert new_g.num_nodes() == g.num_nodes()
    assert F.allclose(g.ndata["h"], new_g.ndata["h"])
    assert "w" in new_g.edata

    # Case2: weighted undirected graph
    g = dgl.graph(([0, 1, 2, 3], [1, 0, 3, 2]), idtype=idtype, device=F.ctx())
    g.edata["w"] = F.tensor([0.1, 0.2, 0.3, 0.4])
    new_g = transform(g)
    src, dst = new_g.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 0), (1, 1), (2, 2), (3, 3)}


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@parametrize_idtype
def test_module_gdc(idtype):
    transform = dgl.GDC([0.1, 0.2, 0.1], avg_degree=1)
    g = dgl.graph(
        ([0, 1, 2, 3, 4], [2, 3, 4, 5, 3]), idtype=idtype, device=F.ctx()
    )
    g.ndata["h"] = F.randn((6, 2))
    new_g = transform(g)
    assert new_g.idtype == g.idtype
    assert new_g.device == g.device
    assert new_g.num_nodes() == g.num_nodes()
    src, dst = new_g.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {
        (0, 0),
        (0, 2),
        (0, 4),
        (1, 1),
        (1, 3),
        (1, 5),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 3),
        (3, 5),
        (4, 3),
        (4, 4),
        (4, 5),
        (5, 5),
    }
    assert F.allclose(g.ndata["h"], new_g.ndata["h"])
    assert "w" in new_g.edata

    # Prior edge weights
    g.edata["w"] = F.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    new_g = transform(g)
    src, dst = new_g.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 0), (1, 1), (2, 2), (3, 3), (4, 3), (4, 4), (5, 5)}


@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow",
    reason="TF doesn't support a slicing operation",
)
@parametrize_idtype
def test_module_node_shuffle(idtype):
    transform = dgl.NodeShuffle()
    g = dgl.heterograph(
        {("A", "r", "B"): ([0, 1], [1, 2])}, idtype=idtype, device=F.ctx()
    )
    g.nodes["B"].data["h"] = F.randn((g.num_nodes("B"), 2))
    old_nfeat = g.nodes["B"].data["h"]
    new_g = transform(g)
    new_nfeat = g.nodes["B"].data["h"]
    assert F.allclose(old_nfeat, new_nfeat)


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@parametrize_idtype
def test_module_drop_node(idtype):
    transform = dgl.DropNode()
    g = dgl.heterograph(
        {("A", "r", "B"): ([0, 1], [1, 2])}, idtype=idtype, device=F.ctx()
    )
    num_nodes_old = g.num_nodes()
    new_g = transform(g)
    assert new_g.idtype == g.idtype
    assert new_g.device == g.device
    assert new_g.ntypes == g.ntypes
    assert new_g.canonical_etypes == g.canonical_etypes
    num_nodes_new = g.num_nodes()
    # Ensure that the original graph is not corrupted
    assert num_nodes_old == num_nodes_new


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@parametrize_idtype
def test_module_drop_edge(idtype):
    transform = dgl.DropEdge()
    g = dgl.heterograph(
        {
            ("A", "r1", "B"): ([0, 1], [1, 2]),
            ("C", "r2", "C"): ([3, 4, 5], [6, 7, 8]),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    num_edges_old = g.num_edges()
    new_g = transform(g)
    assert new_g.idtype == g.idtype
    assert new_g.device == g.device
    assert new_g.ntypes == g.ntypes
    assert new_g.canonical_etypes == g.canonical_etypes
    num_edges_new = g.num_edges()
    # Ensure that the original graph is not corrupted
    assert num_edges_old == num_edges_new


@parametrize_idtype
def test_module_add_edge(idtype):
    transform = dgl.AddEdge()
    g = dgl.heterograph(
        {
            ("A", "r1", "B"): ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5]),
            ("C", "r2", "C"): ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5]),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    num_edges_old = g.num_edges()
    new_g = transform(g)
    assert new_g.num_edges(("A", "r1", "B")) == 6
    assert new_g.num_edges(("C", "r2", "C")) == 6
    assert new_g.idtype == g.idtype
    assert new_g.device == g.device
    assert new_g.ntypes == g.ntypes
    assert new_g.canonical_etypes == g.canonical_etypes
    num_edges_new = g.num_edges()
    # Ensure that the original graph is not corrupted
    assert num_edges_old == num_edges_new


@parametrize_idtype
def test_module_random_walk_pe(idtype):
    transform = dgl.RandomWalkPE(2, "rwpe")
    g = dgl.graph(([0, 1, 1], [1, 1, 0]), idtype=idtype, device=F.ctx())
    new_g = transform(g)
    tgt = F.copy_to(F.tensor([[0.0, 0.5], [0.5, 0.75]]), g.device)
    assert F.allclose(new_g.ndata["rwpe"], tgt)


@parametrize_idtype
def test_module_lap_pe(idtype):
    g = dgl.graph(
        ([2, 1, 0, 3, 1, 1], [3, 1, 1, 2, 1, 0]), idtype=idtype, device=F.ctx()
    )
    tgt_eigval = F.copy_to(
        F.repeat(
            F.tensor([[1.1534e-17, 1.3333e00, 2.0, np.nan, np.nan]]),
            g.num_nodes(),
            dim=0,
        ),
        g.device,
    )
    tgt_pe = F.copy_to(
        F.tensor(
            [
                [0.5, 0.86602539, 0.0, 0.0, 0.0],
                [0.86602539, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.70710677, 0.0, 0.0],
                [0.0, 0.0, 0.70710677, 0.0, 0.0],
            ]
        ),
        g.device,
    )

    # without padding (k<n)
    transform = dgl.LapPE(2, feat_name="lappe")
    new_g = transform(g)
    # tensorflow has no abs() api
    if dgl.backend.backend_name == "tensorflow":
        assert F.allclose(new_g.ndata["lappe"].__abs__(), tgt_pe[:, :2])
    # pytorch & mxnet
    else:
        assert F.allclose(new_g.ndata["lappe"].abs(), tgt_pe[:, :2])

    # with padding (k>=n)
    transform = dgl.LapPE(5, feat_name="lappe", padding=True)
    new_g = transform(g)
    # tensorflow has no abs() api
    if dgl.backend.backend_name == "tensorflow":
        assert F.allclose(new_g.ndata["lappe"].__abs__(), tgt_pe)
    # pytorch & mxnet
    else:
        assert F.allclose(new_g.ndata["lappe"].abs(), tgt_pe)

    # with eigenvalues
    transform = dgl.LapPE(
        5, feat_name="lappe", eigval_name="eigval", padding=True
    )
    new_g = transform(g)
    # tensorflow has no abs() api
    if dgl.backend.backend_name == "tensorflow":
        assert F.allclose(new_g.ndata["eigval"][:, :3], tgt_eigval[:, :3])
        assert F.allclose(new_g.ndata["lappe"].__abs__(), tgt_pe)
    # pytorch & mxnet
    else:
        assert F.allclose(new_g.ndata["eigval"][:, :3], tgt_eigval[:, :3])
        assert F.allclose(new_g.ndata["lappe"].abs(), tgt_pe)


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@pytest.mark.parametrize("g", get_cases(["has_scalar_e_feature"]))
def test_module_sign(g):
    import torch

    atol = 1e-06

    ctx = F.ctx()
    g = g.to(ctx)
    adj = g.adj_external(transpose=True, scipy_fmt="coo").todense()
    adj = torch.tensor(adj).float().to(ctx)

    weight_adj = (
        g.adj_external(transpose=True, scipy_fmt="coo").astype(float).todense()
    )
    weight_adj = torch.tensor(weight_adj).float().to(ctx)
    src, dst = g.edges()
    src, dst = src.long(), dst.long()
    weight_adj[dst, src] = g.edata["scalar_w"]

    # raw
    transform = dgl.SIGNDiffusion(k=1, in_feat_name="h", diffuse_op="raw")
    g = transform(g)
    target = torch.matmul(adj, g.ndata["h"])
    assert torch.allclose(g.ndata["out_feat_1"], target, atol=atol)

    transform = dgl.SIGNDiffusion(
        k=1, in_feat_name="h", eweight_name="scalar_w", diffuse_op="raw"
    )
    g = transform(g)
    target = torch.matmul(weight_adj, g.ndata["h"])
    assert torch.allclose(g.ndata["out_feat_1"], target, atol=atol)

    # rw
    adj_rw = torch.matmul(torch.diag(1 / adj.sum(dim=1)), adj)
    transform = dgl.SIGNDiffusion(k=1, in_feat_name="h", diffuse_op="rw")
    g = transform(g)
    target = torch.matmul(adj_rw, g.ndata["h"])
    assert torch.allclose(g.ndata["out_feat_1"], target, atol=atol)

    weight_adj_rw = torch.matmul(
        torch.diag(1 / weight_adj.sum(dim=1)), weight_adj
    )
    transform = dgl.SIGNDiffusion(
        k=1, in_feat_name="h", eweight_name="scalar_w", diffuse_op="rw"
    )
    g = transform(g)
    target = torch.matmul(weight_adj_rw, g.ndata["h"])
    assert torch.allclose(g.ndata["out_feat_1"], target, atol=atol)

    # gcn
    raw_eweight = g.edata["scalar_w"]
    gcn_norm = dgl.GCNNorm()
    g = gcn_norm(g)
    adj_gcn = adj.clone()
    adj_gcn[dst, src] = g.edata.pop("w")
    transform = dgl.SIGNDiffusion(k=1, in_feat_name="h", diffuse_op="gcn")
    g = transform(g)
    target = torch.matmul(adj_gcn, g.ndata["h"])
    assert torch.allclose(g.ndata["out_feat_1"], target, atol=atol)

    gcn_norm = dgl.GCNNorm("scalar_w")
    g = gcn_norm(g)
    weight_adj_gcn = weight_adj.clone()
    weight_adj_gcn[dst, src] = g.edata["scalar_w"]
    g.edata["scalar_w"] = raw_eweight
    transform = dgl.SIGNDiffusion(
        k=1, in_feat_name="h", eweight_name="scalar_w", diffuse_op="gcn"
    )
    g = transform(g)
    target = torch.matmul(weight_adj_gcn, g.ndata["h"])
    assert torch.allclose(g.ndata["out_feat_1"], target, atol=atol)

    # ppr
    alpha = 0.2
    transform = dgl.SIGNDiffusion(
        k=1, in_feat_name="h", diffuse_op="ppr", alpha=alpha
    )
    g = transform(g)
    target = (1 - alpha) * torch.matmul(
        adj_gcn, g.ndata["h"]
    ) + alpha * g.ndata["h"]
    assert torch.allclose(g.ndata["out_feat_1"], target, atol=atol)

    transform = dgl.SIGNDiffusion(
        k=1,
        in_feat_name="h",
        eweight_name="scalar_w",
        diffuse_op="ppr",
        alpha=alpha,
    )
    g = transform(g)
    target = (1 - alpha) * torch.matmul(
        weight_adj_gcn, g.ndata["h"]
    ) + alpha * g.ndata["h"]
    assert torch.allclose(g.ndata["out_feat_1"], target, atol=atol)


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@parametrize_idtype
def test_module_row_feat_normalizer(idtype):
    # Case1: Normalize features of a homogeneous graph.
    transform = dgl.RowFeatNormalizer(
        subtract_min=True, node_feat_names=["h"], edge_feat_names=["w"]
    )
    g = dgl.rand_graph(5, 5, idtype=idtype, device=F.ctx())
    g.ndata["h"] = F.randn((g.num_nodes(), 128))
    g.edata["w"] = F.randn((g.num_edges(), 128))
    g = transform(g)
    assert g.ndata["h"].shape == (g.num_nodes(), 128)
    assert g.edata["w"].shape == (g.num_edges(), 128)
    assert F.allclose(g.ndata["h"].sum(1), F.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))
    assert F.allclose(g.edata["w"].sum(1), F.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))

    # Case2: Normalize features of a heterogeneous graph.
    transform = dgl.RowFeatNormalizer(
        subtract_min=True, node_feat_names=["h", "h2"], edge_feat_names=["w"]
    )
    g = dgl.heterograph(
        {
            ("user", "follows", "user"): (F.tensor([1, 2]), F.tensor([3, 4])),
            ("player", "plays", "game"): (F.tensor([2, 2]), F.tensor([1, 1])),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    g.ndata["h"] = {"game": F.randn((2, 128)), "player": F.randn((3, 128))}
    g.ndata["h2"] = {"user": F.randn((5, 128))}
    g.edata["w"] = {
        ("user", "follows", "user"): F.randn((2, 128)),
        ("player", "plays", "game"): F.randn((2, 128)),
    }
    g = transform(g)
    assert g.ndata["h"]["game"].shape == (2, 128)
    assert g.ndata["h"]["player"].shape == (3, 128)
    assert g.ndata["h2"]["user"].shape == (5, 128)
    assert g.edata["w"][("user", "follows", "user")].shape == (2, 128)
    assert g.edata["w"][("player", "plays", "game")].shape == (2, 128)
    assert F.allclose(g.ndata["h"]["game"].sum(1), F.tensor([1.0, 1.0]))
    assert F.allclose(g.ndata["h"]["player"].sum(1), F.tensor([1.0, 1.0, 1.0]))
    assert F.allclose(
        g.ndata["h2"]["user"].sum(1), F.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    )
    assert F.allclose(
        g.edata["w"][("user", "follows", "user")].sum(1), F.tensor([1.0, 1.0])
    )
    assert F.allclose(
        g.edata["w"][("player", "plays", "game")].sum(1), F.tensor([1.0, 1.0])
    )


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@parametrize_idtype
def test_module_feat_mask(idtype):
    # Case1: Mask node and edge feature tensors of a homogeneous graph.
    transform = dgl.FeatMask(node_feat_names=["h"], edge_feat_names=["w"])
    g = dgl.rand_graph(5, 20, idtype=idtype, device=F.ctx())
    g.ndata["h"] = F.ones((g.num_nodes(), 10))
    g.edata["w"] = F.ones((g.num_edges(), 20))
    g = transform(g)
    assert g.device == g.device
    assert g.idtype == g.idtype
    assert g.ndata["h"].shape == (g.num_nodes(), 10)
    assert g.edata["w"].shape == (g.num_edges(), 20)

    # Case2: Mask node and edge feature tensors of a heterogeneous graph.
    g = dgl.heterograph(
        {
            ("user", "follows", "user"): (F.tensor([1, 2]), F.tensor([3, 4])),
            ("player", "plays", "game"): (F.tensor([2, 2]), F.tensor([1, 1])),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    g.ndata["h"] = {"game": F.randn((2, 5)), "player": F.randn((3, 5))}
    g.edata["w"] = {
        ("user", "follows", "user"): F.randn((2, 5)),
        ("player", "plays", "game"): F.randn((2, 5)),
    }
    g = transform(g)
    assert g.device == g.device
    assert g.idtype == g.idtype
    assert g.ndata["h"]["game"].shape == (2, 5)
    assert g.ndata["h"]["player"].shape == (3, 5)
    assert g.edata["w"][("user", "follows", "user")].shape == (2, 5)
    assert g.edata["w"][("player", "plays", "game")].shape == (2, 5)


@parametrize_idtype
def test_shortest_dist(idtype):
    g = dgl.graph(([0, 1, 1, 2], [2, 0, 3, 3]), idtype=idtype, device=F.ctx())

    # case 1: directed single source
    dist = dgl.shortest_dist(g, root=0)
    tgt = F.copy_to(F.tensor([0, -1, 1, 2], dtype=F.int64), g.device)
    assert F.array_equal(dist, tgt)

    # case 2: undirected all pairs
    dist, paths = dgl.shortest_dist(g, root=None, return_paths=True)
    tgt_dist = F.copy_to(
        F.tensor(
            [[0, -1, 1, 2], [1, 0, 2, 1], [-1, -1, 0, 1], [-1, -1, -1, 0]],
            dtype=F.int64,
        ),
        g.device,
    )
    tgt_paths = F.copy_to(
        F.tensor(
            [
                [[-1, -1], [-1, -1], [0, -1], [0, 3]],
                [[1, -1], [-1, -1], [1, 0], [2, -1]],
                [[-1, -1], [-1, -1], [-1, -1], [3, -1]],
                [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],
            ],
            dtype=F.int64,
        ),
        g.device,
    )
    assert F.array_equal(dist, tgt_dist)
    assert F.array_equal(paths, tgt_paths)


@parametrize_idtype
def test_module_to_levi(idtype):
    transform = dgl.ToLevi()
    g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 0]), idtype=idtype, device=F.ctx())
    g.ndata["h"] = F.randn((g.num_nodes(), 2))
    g.edata["w"] = F.randn((g.num_edges(), 2))
    lg = transform(g)
    assert lg.device == g.device
    assert lg.idtype == g.idtype
    assert lg.ntypes == ["edge", "node"]
    assert lg.canonical_etypes == [
        ("edge", "e2n", "node"),
        ("node", "n2e", "edge"),
    ]
    assert lg.num_nodes("node") == g.num_nodes()
    assert lg.num_nodes("edge") == g.num_edges()
    assert lg.num_edges("n2e") == g.num_edges()
    assert lg.num_edges("e2n") == g.num_edges()

    src, dst = lg.edges(etype="n2e")
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 0), (1, 1), (2, 2), (3, 3)}

    src, dst = lg.edges(etype="e2n")
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == {(0, 1), (1, 2), (2, 3), (3, 0)}

    assert F.allclose(lg.nodes["node"].data["h"], g.ndata["h"])
    assert F.allclose(lg.nodes["edge"].data["w"], g.edata["w"])


@parametrize_idtype
def test_module_svd_pe(idtype):
    g = dgl.graph(
        (
            [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4],
            [2, 3, 0, 2, 0, 2, 3, 4, 3, 4, 0, 1],
        ),
        idtype=idtype,
        device=F.ctx(),
    )
    # without padding
    tgt_pe = F.copy_to(
        F.tensor(
            [
                [0.6669, 0.3068, 0.7979, 0.8477],
                [0.6311, 0.6101, 0.1248, 0.5137],
                [1.1993, 0.0665, 0.9183, 0.1455],
                [0.5682, 0.6766, 0.8952, 0.6449],
                [0.3393, 0.8363, 0.6500, 0.4564],
            ]
        ),
        g.device,
    )
    transform_1 = dgl.SVDPE(k=2, feat_name="svd_pe")
    g1 = transform_1(g)
    if dgl.backend.backend_name == "tensorflow":
        assert F.allclose(g1.ndata["svd_pe"].__abs__(), tgt_pe)
    else:
        assert F.allclose(g1.ndata["svd_pe"].abs(), tgt_pe)

    # with padding
    transform_2 = dgl.SVDPE(k=6, feat_name="svd_pe", padding=True)
    g2 = transform_2(g)
    assert F.shape(g2.ndata["svd_pe"]) == (5, 12)


if __name__ == "__main__":
    test_partition_with_halo()
    test_module_heat_kernel(F.int32)
