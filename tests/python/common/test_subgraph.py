import unittest

import backend as F

import dgl
import networkx as nx
import numpy as np
import pytest
import scipy.sparse as ssp
from utils import parametrize_idtype

D = 5


def generate_graph(grad=False, add_data=True):
    g = dgl.graph([]).to(F.ctx())
    g.add_nodes(10)
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edges(0, i)
        g.add_edges(i, 9)
    # add a back flow from 9 to 0
    g.add_edges(9, 0)
    if add_data:
        ncol = F.randn((10, D))
        ecol = F.randn((17, D))
        if grad:
            ncol = F.attach_grad(ncol)
            ecol = F.attach_grad(ecol)
        g.ndata["h"] = ncol
        g.edata["l"] = ecol
    return g


def test_edge_subgraph():
    # Test when the graph has no node data and edge data.
    g = generate_graph(add_data=False)
    eid = [0, 2, 3, 6, 7, 9]

    # relabel=True
    sg = g.edge_subgraph(eid)
    assert F.array_equal(
        sg.ndata[dgl.NID], F.tensor([0, 2, 4, 5, 1, 9], g.idtype)
    )
    assert F.array_equal(sg.edata[dgl.EID], F.tensor(eid, g.idtype))
    sg.ndata["h"] = F.arange(0, sg.num_nodes())
    sg.edata["h"] = F.arange(0, sg.num_edges())

    # relabel=False
    sg = g.edge_subgraph(eid, relabel_nodes=False)
    assert g.num_nodes() == sg.num_nodes()
    assert F.array_equal(sg.edata[dgl.EID], F.tensor(eid, g.idtype))
    sg.ndata["h"] = F.arange(0, sg.num_nodes())
    sg.edata["h"] = F.arange(0, sg.num_edges())


@pytest.mark.parametrize("relabel_nodes", [True, False])
def test_subgraph_relabel_nodes(relabel_nodes):
    g = generate_graph()
    h = g.ndata["h"]
    l = g.edata["l"]
    nid = [0, 2, 3, 6, 7, 9]
    sg = g.subgraph(nid, relabel_nodes=relabel_nodes)
    eid = {2, 3, 4, 5, 10, 11, 12, 13, 16}
    assert set(F.asnumpy(sg.edata[dgl.EID])) == eid
    eid = sg.edata[dgl.EID]
    # the subgraph is empty initially except for EID field
    # the subgraph is empty initially except for NID field if relabel_nodes
    if relabel_nodes:
        assert len(sg.ndata) == 2
    assert len(sg.edata) == 2
    sh = sg.ndata["h"]
    # The node number is not reduced if relabel_node=False.
    # The subgraph keeps the same node information as the original graph.
    if relabel_nodes:
        assert F.allclose(F.gather_row(h, F.tensor(nid)), sh)
    else:
        assert F.allclose(
            F.gather_row(h, F.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])), sh
        )
    # The s,d,eid means the source node, destination node and edge id of the subgraph.
    # The edges labeled 1 are those selected by the subgraph.
    """
    s, d, eid
    0, 1, 0
    1, 9, 1
    0, 2, 2    1
    2, 9, 3    1
    0, 3, 4    1
    3, 9, 5    1
    0, 4, 6
    4, 9, 7
    0, 5, 8
    5, 9, 9       3
    0, 6, 10   1
    6, 9, 11   1  3
    0, 7, 12   1
    7, 9, 13   1  3
    0, 8, 14
    8, 9, 15      3
    9, 0, 16   1
    """
    assert F.allclose(F.gather_row(l, eid), sg.edata["l"])
    # update the node/edge features on the subgraph should NOT
    # reflect to the parent graph.
    if relabel_nodes:
        sg.ndata["h"] = F.zeros((6, D))
    else:
        sg.ndata["h"] = F.zeros((10, D))
    assert F.allclose(h, g.ndata["h"])


def _test_map_to_subgraph():
    g = dgl.graph([])
    g.add_nodes(10)
    g.add_edges(F.arange(0, 9), F.arange(1, 10))
    h = g.subgraph([0, 1, 2, 5, 8])
    v = h.map_to_subgraph_nid([0, 8, 2])
    assert np.array_equal(F.asnumpy(v), np.array([0, 4, 2]))


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
            ("user", "follows", "user"): ([0, 1], [1, 2]),
            ("user", "plays", "game"): ([0, 1, 2, 1], [0, 0, 1, 1]),
            ("user", "wishes", "game"): ([0, 2], [1, 0]),
            ("developer", "develops", "game"): ([0, 1], [0, 1]),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    for etype in g.etypes:
        g.edges[etype].data["weight"] = F.randn((g.num_edges(etype),))
    assert g.idtype == idtype
    assert g.device == F.ctx()
    return g


def create_test_heterograph2(idtype):
    """test heterograph from the docstring, with an empty relation"""
    # 3 users, 2 games, 2 developers
    # metagraph:
    #    ('user', 'follows', 'user'),
    #    ('user', 'plays', 'game'),
    #    ('user', 'wishes', 'game'),
    #    ('developer', 'develops', 'game')

    g = dgl.heterograph(
        {
            ("user", "follows", "user"): ([0, 1], [1, 2]),
            ("user", "plays", "game"): ([0, 1, 2, 1], [0, 0, 1, 1]),
            ("user", "wishes", "game"): ([0, 2], [1, 0]),
            ("developer", "develops", "game"): ([], []),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    for etype in g.etypes:
        g.edges[etype].data["weight"] = F.randn((g.num_edges(etype),))
    assert g.idtype == idtype
    assert g.device == F.ctx()
    return g


@unittest.skipIf(
    dgl.backend.backend_name == "mxnet",
    reason="MXNet doesn't support bool tensor",
)
@parametrize_idtype
def test_subgraph_mask(idtype):
    g = create_test_heterograph(idtype)
    g_graph = g["follows"]
    g_bipartite = g["plays"]

    x = F.randn((3, 5))
    y = F.randn((2, 4))
    g.nodes["user"].data["h"] = x
    g.edges["follows"].data["h"] = y

    def _check_subgraph(g, sg):
        assert sg.idtype == g.idtype
        assert sg.device == g.device
        assert sg.ntypes == g.ntypes
        assert sg.etypes == g.etypes
        assert sg.canonical_etypes == g.canonical_etypes
        assert F.array_equal(
            F.tensor(sg.nodes["user"].data[dgl.NID]), F.tensor([1, 2], idtype)
        )
        assert F.array_equal(
            F.tensor(sg.nodes["game"].data[dgl.NID]), F.tensor([0], idtype)
        )
        assert F.array_equal(
            F.tensor(sg.edges["follows"].data[dgl.EID]), F.tensor([1], idtype)
        )
        assert F.array_equal(
            F.tensor(sg.edges["plays"].data[dgl.EID]), F.tensor([1], idtype)
        )
        assert F.array_equal(
            F.tensor(sg.edges["wishes"].data[dgl.EID]), F.tensor([1], idtype)
        )
        assert sg.num_nodes("developer") == 0
        assert sg.num_edges("develops") == 0
        assert F.array_equal(
            sg.nodes["user"].data["h"], g.nodes["user"].data["h"][1:3]
        )
        assert F.array_equal(
            sg.edges["follows"].data["h"], g.edges["follows"].data["h"][1:2]
        )

    sg1 = g.subgraph(
        {
            "user": F.tensor([False, True, True], dtype=F.bool),
            "game": F.tensor([True, False, False, False], dtype=F.bool),
        }
    )
    _check_subgraph(g, sg1)
    sg2 = g.edge_subgraph(
        {
            "follows": F.tensor([False, True], dtype=F.bool),
            "plays": F.tensor([False, True, False, False], dtype=F.bool),
            "wishes": F.tensor([False, True], dtype=F.bool),
        }
    )
    _check_subgraph(g, sg2)


@parametrize_idtype
def test_subgraph1(idtype):
    g = create_test_heterograph(idtype)
    g_graph = g["follows"]
    g_bipartite = g["plays"]

    x = F.randn((3, 5))
    y = F.randn((2, 4))
    g.nodes["user"].data["h"] = x
    g.edges["follows"].data["h"] = y

    def _check_subgraph(g, sg):
        assert sg.idtype == g.idtype
        assert sg.device == g.device
        assert sg.ntypes == g.ntypes
        assert sg.etypes == g.etypes
        assert sg.canonical_etypes == g.canonical_etypes
        assert F.array_equal(
            F.tensor(sg.nodes["user"].data[dgl.NID]), F.tensor([1, 2], g.idtype)
        )
        assert F.array_equal(
            F.tensor(sg.nodes["game"].data[dgl.NID]), F.tensor([0], g.idtype)
        )
        assert F.array_equal(
            F.tensor(sg.edges["follows"].data[dgl.EID]), F.tensor([1], g.idtype)
        )
        assert F.array_equal(
            F.tensor(sg.edges["plays"].data[dgl.EID]), F.tensor([1], g.idtype)
        )
        assert F.array_equal(
            F.tensor(sg.edges["wishes"].data[dgl.EID]), F.tensor([1], g.idtype)
        )
        assert sg.num_nodes("developer") == 0
        assert sg.num_edges("develops") == 0
        assert F.array_equal(
            sg.nodes["user"].data["h"], g.nodes["user"].data["h"][1:3]
        )
        assert F.array_equal(
            sg.edges["follows"].data["h"], g.edges["follows"].data["h"][1:2]
        )

    sg1 = g.subgraph({"user": [1, 2], "game": [0]})
    _check_subgraph(g, sg1)
    sg2 = g.edge_subgraph({"follows": [1], "plays": [1], "wishes": [1]})
    _check_subgraph(g, sg2)

    # backend tensor input
    sg1 = g.subgraph(
        {
            "user": F.tensor([1, 2], dtype=idtype),
            "game": F.tensor([0], dtype=idtype),
        }
    )
    _check_subgraph(g, sg1)
    sg2 = g.edge_subgraph(
        {
            "follows": F.tensor([1], dtype=idtype),
            "plays": F.tensor([1], dtype=idtype),
            "wishes": F.tensor([1], dtype=idtype),
        }
    )
    _check_subgraph(g, sg2)

    # numpy input
    sg1 = g.subgraph({"user": np.array([1, 2]), "game": np.array([0])})
    _check_subgraph(g, sg1)
    sg2 = g.edge_subgraph(
        {
            "follows": np.array([1]),
            "plays": np.array([1]),
            "wishes": np.array([1]),
        }
    )
    _check_subgraph(g, sg2)

    def _check_subgraph_single_ntype(g, sg, preserve_nodes=False):
        assert sg.idtype == g.idtype
        assert sg.device == g.device
        assert sg.ntypes == g.ntypes
        assert sg.etypes == g.etypes
        assert sg.canonical_etypes == g.canonical_etypes

        if not preserve_nodes:
            assert F.array_equal(
                F.tensor(sg.nodes["user"].data[dgl.NID]),
                F.tensor([1, 2], g.idtype),
            )
        else:
            for ntype in sg.ntypes:
                assert g.num_nodes(ntype) == sg.num_nodes(ntype)

        assert F.array_equal(
            F.tensor(sg.edges["follows"].data[dgl.EID]), F.tensor([1], g.idtype)
        )

        if not preserve_nodes:
            assert F.array_equal(
                sg.nodes["user"].data["h"], g.nodes["user"].data["h"][1:3]
            )
        assert F.array_equal(
            sg.edges["follows"].data["h"], g.edges["follows"].data["h"][1:2]
        )

    def _check_subgraph_single_etype(g, sg, preserve_nodes=False):
        assert sg.ntypes == g.ntypes
        assert sg.etypes == g.etypes
        assert sg.canonical_etypes == g.canonical_etypes

        if not preserve_nodes:
            assert F.array_equal(
                F.tensor(sg.nodes["user"].data[dgl.NID]),
                F.tensor([0, 1], g.idtype),
            )
            assert F.array_equal(
                F.tensor(sg.nodes["game"].data[dgl.NID]),
                F.tensor([0], g.idtype),
            )
        else:
            for ntype in sg.ntypes:
                assert g.num_nodes(ntype) == sg.num_nodes(ntype)

        assert F.array_equal(
            F.tensor(sg.edges["plays"].data[dgl.EID]),
            F.tensor([0, 1], g.idtype),
        )

    sg1_graph = g_graph.subgraph([1, 2])
    _check_subgraph_single_ntype(g_graph, sg1_graph)
    sg1_graph = g_graph.edge_subgraph([1])
    _check_subgraph_single_ntype(g_graph, sg1_graph)
    sg1_graph = g_graph.edge_subgraph([1], relabel_nodes=False)
    _check_subgraph_single_ntype(g_graph, sg1_graph, True)
    sg2_bipartite = g_bipartite.edge_subgraph([0, 1])
    _check_subgraph_single_etype(g_bipartite, sg2_bipartite)
    sg2_bipartite = g_bipartite.edge_subgraph([0, 1], relabel_nodes=False)
    _check_subgraph_single_etype(g_bipartite, sg2_bipartite, True)

    def _check_typed_subgraph1(g, sg):
        assert g.idtype == sg.idtype
        assert g.device == sg.device
        assert set(sg.ntypes) == {"user", "game"}
        assert set(sg.etypes) == {"follows", "plays", "wishes"}
        for ntype in sg.ntypes:
            assert sg.num_nodes(ntype) == g.num_nodes(ntype)
        for etype in sg.etypes:
            src_sg, dst_sg = sg.all_edges(etype=etype, order="eid")
            src_g, dst_g = g.all_edges(etype=etype, order="eid")
            assert F.array_equal(src_sg, src_g)
            assert F.array_equal(dst_sg, dst_g)
        assert F.array_equal(
            sg.nodes["user"].data["h"], g.nodes["user"].data["h"]
        )
        assert F.array_equal(
            sg.edges["follows"].data["h"], g.edges["follows"].data["h"]
        )
        g.nodes["user"].data["h"] = F.scatter_row(
            g.nodes["user"].data["h"], F.tensor([2]), F.randn((1, 5))
        )
        g.edges["follows"].data["h"] = F.scatter_row(
            g.edges["follows"].data["h"], F.tensor([1]), F.randn((1, 4))
        )
        assert F.array_equal(
            sg.nodes["user"].data["h"], g.nodes["user"].data["h"]
        )
        assert F.array_equal(
            sg.edges["follows"].data["h"], g.edges["follows"].data["h"]
        )

    def _check_typed_subgraph2(g, sg):
        assert set(sg.ntypes) == {"developer", "game"}
        assert set(sg.etypes) == {"develops"}
        for ntype in sg.ntypes:
            assert sg.num_nodes(ntype) == g.num_nodes(ntype)
        for etype in sg.etypes:
            src_sg, dst_sg = sg.all_edges(etype=etype, order="eid")
            src_g, dst_g = g.all_edges(etype=etype, order="eid")
            assert F.array_equal(src_sg, src_g)
            assert F.array_equal(dst_sg, dst_g)

    sg3 = g.node_type_subgraph(["user", "game"])
    _check_typed_subgraph1(g, sg3)
    sg4 = g.edge_type_subgraph(["develops"])
    _check_typed_subgraph2(g, sg4)
    sg5 = g.edge_type_subgraph(["follows", "plays", "wishes"])
    _check_typed_subgraph1(g, sg5)

    # Test for restricted format
    for fmt in ["csr", "csc", "coo"]:
        g = dgl.graph(([0, 1], [1, 2])).formats(fmt)
        sg = g.subgraph({g.ntypes[0]: [1, 0]})
        nids = F.asnumpy(sg.ndata[dgl.NID])
        assert np.array_equal(nids, np.array([1, 0]))
        src, dst = sg.edges(order="eid")
        src = F.asnumpy(src)
        dst = F.asnumpy(dst)
        assert np.array_equal(src, np.array([1]))


@parametrize_idtype
def test_in_subgraph(idtype):
    hg = dgl.heterograph(
        {
            ("user", "follow", "user"): (
                [1, 2, 3, 0, 2, 3, 0],
                [0, 0, 0, 1, 1, 1, 2],
            ),
            ("user", "play", "game"): ([0, 0, 1, 3], [0, 1, 2, 2]),
            ("game", "liked-by", "user"): (
                [2, 2, 2, 1, 1, 0],
                [0, 1, 2, 0, 3, 0],
            ),
            ("user", "flips", "coin"): ([0, 1, 2, 3], [0, 0, 0, 0]),
        },
        idtype=idtype,
        num_nodes_dict={"user": 5, "game": 10, "coin": 8},
    ).to(F.ctx())
    subg = dgl.in_subgraph(hg, {"user": [0, 1], "game": 0})
    assert subg.idtype == idtype
    assert len(subg.ntypes) == 3
    assert len(subg.etypes) == 4
    u, v = subg["follow"].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert F.array_equal(
        hg["follow"].edge_ids(u, v), subg["follow"].edata[dgl.EID]
    )
    assert edge_set == {(1, 0), (2, 0), (3, 0), (0, 1), (2, 1), (3, 1)}
    u, v = subg["play"].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert F.array_equal(hg["play"].edge_ids(u, v), subg["play"].edata[dgl.EID])
    assert edge_set == {(0, 0)}
    u, v = subg["liked-by"].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert F.array_equal(
        hg["liked-by"].edge_ids(u, v), subg["liked-by"].edata[dgl.EID]
    )
    assert edge_set == {(2, 0), (2, 1), (1, 0), (0, 0)}
    assert subg["flips"].num_edges() == 0
    for ntype in subg.ntypes:
        assert dgl.NID not in subg.nodes[ntype].data

    # Test store_ids
    subg = dgl.in_subgraph(hg, {"user": [0, 1], "game": 0}, store_ids=False)
    for etype in ["follow", "play", "liked-by"]:
        assert dgl.EID not in subg.edges[etype].data
    for ntype in subg.ntypes:
        assert dgl.NID not in subg.nodes[ntype].data

    # Test relabel nodes
    subg = dgl.in_subgraph(hg, {"user": [0, 1], "game": 0}, relabel_nodes=True)
    assert subg.idtype == idtype
    assert len(subg.ntypes) == 3
    assert len(subg.etypes) == 4

    u, v = subg["follow"].edges()
    old_u = F.gather_row(subg.nodes["user"].data[dgl.NID], u)
    old_v = F.gather_row(subg.nodes["user"].data[dgl.NID], v)
    assert F.array_equal(
        hg["follow"].edge_ids(old_u, old_v), subg["follow"].edata[dgl.EID]
    )
    edge_set = set(zip(list(F.asnumpy(old_u)), list(F.asnumpy(old_v))))
    assert edge_set == {(1, 0), (2, 0), (3, 0), (0, 1), (2, 1), (3, 1)}

    u, v = subg["play"].edges()
    old_u = F.gather_row(subg.nodes["user"].data[dgl.NID], u)
    old_v = F.gather_row(subg.nodes["game"].data[dgl.NID], v)
    assert F.array_equal(
        hg["play"].edge_ids(old_u, old_v), subg["play"].edata[dgl.EID]
    )
    edge_set = set(zip(list(F.asnumpy(old_u)), list(F.asnumpy(old_v))))
    assert edge_set == {(0, 0)}

    u, v = subg["liked-by"].edges()
    old_u = F.gather_row(subg.nodes["game"].data[dgl.NID], u)
    old_v = F.gather_row(subg.nodes["user"].data[dgl.NID], v)
    assert F.array_equal(
        hg["liked-by"].edge_ids(old_u, old_v), subg["liked-by"].edata[dgl.EID]
    )
    edge_set = set(zip(list(F.asnumpy(old_u)), list(F.asnumpy(old_v))))
    assert edge_set == {(2, 0), (2, 1), (1, 0), (0, 0)}

    assert subg.num_nodes("user") == 4
    assert subg.num_nodes("game") == 3
    assert subg.num_nodes("coin") == 0
    assert subg.num_edges("flips") == 0


@parametrize_idtype
def test_out_subgraph(idtype):
    hg = dgl.heterograph(
        {
            ("user", "follow", "user"): (
                [1, 2, 3, 0, 2, 3, 0],
                [0, 0, 0, 1, 1, 1, 2],
            ),
            ("user", "play", "game"): ([0, 0, 1, 3], [0, 1, 2, 2]),
            ("game", "liked-by", "user"): (
                [2, 2, 2, 1, 1, 0],
                [0, 1, 2, 0, 3, 0],
            ),
            ("user", "flips", "coin"): ([0, 1, 2, 3], [0, 0, 0, 0]),
        },
        idtype=idtype,
    ).to(F.ctx())
    subg = dgl.out_subgraph(hg, {"user": [0, 1], "game": 0})
    assert subg.idtype == idtype
    assert len(subg.ntypes) == 3
    assert len(subg.etypes) == 4
    u, v = subg["follow"].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert edge_set == {(1, 0), (0, 1), (0, 2)}
    assert F.array_equal(
        hg["follow"].edge_ids(u, v), subg["follow"].edata[dgl.EID]
    )
    u, v = subg["play"].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert edge_set == {(0, 0), (0, 1), (1, 2)}
    assert F.array_equal(hg["play"].edge_ids(u, v), subg["play"].edata[dgl.EID])
    u, v = subg["liked-by"].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert edge_set == {(0, 0)}
    assert F.array_equal(
        hg["liked-by"].edge_ids(u, v), subg["liked-by"].edata[dgl.EID]
    )
    u, v = subg["flips"].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert edge_set == {(0, 0), (1, 0)}
    assert F.array_equal(
        hg["flips"].edge_ids(u, v), subg["flips"].edata[dgl.EID]
    )
    for ntype in subg.ntypes:
        assert dgl.NID not in subg.nodes[ntype].data

    # Test store_ids
    subg = dgl.out_subgraph(hg, {"user": [0, 1], "game": 0}, store_ids=False)
    for etype in subg.canonical_etypes:
        assert dgl.EID not in subg.edges[etype].data
    for ntype in subg.ntypes:
        assert dgl.NID not in subg.nodes[ntype].data

    # Test relabel nodes
    subg = dgl.out_subgraph(hg, {"user": [1], "game": 0}, relabel_nodes=True)
    assert subg.idtype == idtype
    assert len(subg.ntypes) == 3
    assert len(subg.etypes) == 4

    u, v = subg["follow"].edges()
    old_u = F.gather_row(subg.nodes["user"].data[dgl.NID], u)
    old_v = F.gather_row(subg.nodes["user"].data[dgl.NID], v)
    edge_set = set(zip(list(F.asnumpy(old_u)), list(F.asnumpy(old_v))))
    assert edge_set == {(1, 0)}
    assert F.array_equal(
        hg["follow"].edge_ids(old_u, old_v), subg["follow"].edata[dgl.EID]
    )

    u, v = subg["play"].edges()
    old_u = F.gather_row(subg.nodes["user"].data[dgl.NID], u)
    old_v = F.gather_row(subg.nodes["game"].data[dgl.NID], v)
    edge_set = set(zip(list(F.asnumpy(old_u)), list(F.asnumpy(old_v))))
    assert edge_set == {(1, 2)}
    assert F.array_equal(
        hg["play"].edge_ids(old_u, old_v), subg["play"].edata[dgl.EID]
    )

    u, v = subg["liked-by"].edges()
    old_u = F.gather_row(subg.nodes["game"].data[dgl.NID], u)
    old_v = F.gather_row(subg.nodes["user"].data[dgl.NID], v)
    edge_set = set(zip(list(F.asnumpy(old_u)), list(F.asnumpy(old_v))))
    assert edge_set == {(0, 0)}
    assert F.array_equal(
        hg["liked-by"].edge_ids(old_u, old_v), subg["liked-by"].edata[dgl.EID]
    )

    u, v = subg["flips"].edges()
    old_u = F.gather_row(subg.nodes["user"].data[dgl.NID], u)
    old_v = F.gather_row(subg.nodes["coin"].data[dgl.NID], v)
    edge_set = set(zip(list(F.asnumpy(old_u)), list(F.asnumpy(old_v))))
    assert edge_set == {(1, 0)}
    assert F.array_equal(
        hg["flips"].edge_ids(old_u, old_v), subg["flips"].edata[dgl.EID]
    )
    assert subg.num_nodes("user") == 2
    assert subg.num_nodes("game") == 2
    assert subg.num_nodes("coin") == 1


def test_subgraph_message_passing():
    # Unit test for PR #2055
    g = dgl.graph(([0, 1, 2], [2, 3, 4])).to(F.cpu())
    g.ndata["x"] = F.copy_to(F.randn((5, 6)), F.cpu())
    sg = g.subgraph([1, 2, 3]).to(F.ctx())
    sg.update_all(
        lambda edges: {"x": edges.src["x"]},
        lambda nodes: {"y": F.sum(nodes.mailbox["x"], 1)},
    )


@parametrize_idtype
def test_khop_in_subgraph(idtype):
    g = dgl.graph(
        ([1, 1, 2, 3, 4], [0, 2, 0, 4, 2]), idtype=idtype, device=F.ctx()
    )
    g.edata["w"] = F.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    sg, inv = dgl.khop_in_subgraph(g, 0, k=2)
    assert sg.idtype == g.idtype
    u, v = sg.edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert edge_set == {(1, 0), (1, 2), (2, 0), (3, 2)}
    assert F.array_equal(
        sg.edata[dgl.EID], F.tensor([0, 1, 2, 4], dtype=idtype)
    )
    assert F.array_equal(
        sg.edata["w"], F.tensor([[0, 1], [2, 3], [4, 5], [8, 9]])
    )
    assert F.array_equal(F.astype(inv, idtype), F.tensor([0], idtype))

    # Test multiple nodes
    sg, inv = dgl.khop_in_subgraph(g, [0, 2], k=1)
    assert sg.num_edges() == 4

    sg, inv = dgl.khop_in_subgraph(g, F.tensor([0, 2], idtype), k=1)
    assert sg.num_edges() == 4

    # Test isolated node
    sg, inv = dgl.khop_in_subgraph(g, 1, k=2)
    assert sg.idtype == g.idtype
    assert sg.num_nodes() == 1
    assert sg.num_edges() == 0
    assert F.array_equal(F.astype(inv, idtype), F.tensor([0], idtype))

    g = dgl.heterograph(
        {
            ("user", "plays", "game"): ([0, 1, 1, 2], [0, 0, 2, 1]),
            ("user", "follows", "user"): ([0, 1, 1], [1, 2, 2]),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    sg, inv = dgl.khop_in_subgraph(g, {"game": 0}, k=2)
    assert sg.idtype == idtype
    assert sg.num_nodes("game") == 1
    assert sg.num_nodes("user") == 2
    assert len(sg.ntypes) == 2
    assert len(sg.etypes) == 2
    u, v = sg["follows"].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert edge_set == {(0, 1)}
    u, v = sg["plays"].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert edge_set == {(0, 0), (1, 0)}
    assert F.array_equal(F.astype(inv["game"], idtype), F.tensor([0], idtype))

    # Test isolated node
    sg, inv = dgl.khop_in_subgraph(g, {"user": 0}, k=2)
    assert sg.idtype == idtype
    assert sg.num_nodes("game") == 0
    assert sg.num_nodes("user") == 1
    assert sg.num_edges("follows") == 0
    assert sg.num_edges("plays") == 0
    assert F.array_equal(F.astype(inv["user"], idtype), F.tensor([0], idtype))

    # Test multiple nodes
    sg, inv = dgl.khop_in_subgraph(
        g, {"user": F.tensor([0, 1], idtype), "game": 0}, k=1
    )
    u, v = sg["follows"].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert edge_set == {(0, 1)}
    u, v = sg["plays"].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert edge_set == {(0, 0), (1, 0)}
    assert F.array_equal(
        F.astype(inv["user"], idtype), F.tensor([0, 1], idtype)
    )
    assert F.array_equal(F.astype(inv["game"], idtype), F.tensor([0], idtype))


@parametrize_idtype
def test_khop_out_subgraph(idtype):
    g = dgl.graph(
        ([0, 2, 0, 4, 2], [1, 1, 2, 3, 4]), idtype=idtype, device=F.ctx()
    )
    g.edata["w"] = F.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    sg, inv = dgl.khop_out_subgraph(g, 0, k=2)
    assert sg.idtype == g.idtype
    u, v = sg.edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert edge_set == {(0, 1), (2, 1), (0, 2), (2, 3)}
    assert F.array_equal(
        sg.edata[dgl.EID], F.tensor([0, 2, 1, 4], dtype=idtype)
    )
    assert F.array_equal(
        sg.edata["w"], F.tensor([[0, 1], [4, 5], [2, 3], [8, 9]])
    )
    assert F.array_equal(F.astype(inv, idtype), F.tensor([0], idtype))

    # Test multiple nodes
    sg, inv = dgl.khop_out_subgraph(g, [0, 2], k=1)
    assert sg.num_edges() == 4

    sg, inv = dgl.khop_out_subgraph(g, F.tensor([0, 2], idtype), k=1)
    assert sg.num_edges() == 4

    # Test isolated node
    sg, inv = dgl.khop_out_subgraph(g, 1, k=2)
    assert sg.idtype == g.idtype
    assert sg.num_nodes() == 1
    assert sg.num_edges() == 0
    assert F.array_equal(F.astype(inv, idtype), F.tensor([0], idtype))

    g = dgl.heterograph(
        {
            ("user", "plays", "game"): ([0, 1, 1, 2], [0, 0, 2, 1]),
            ("user", "follows", "user"): ([0, 1], [1, 3]),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    sg, inv = dgl.khop_out_subgraph(g, {"user": 0}, k=2)
    assert sg.idtype == idtype
    assert sg.num_nodes("game") == 2
    assert sg.num_nodes("user") == 3
    assert len(sg.ntypes) == 2
    assert len(sg.etypes) == 2
    u, v = sg["follows"].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert edge_set == {(0, 1), (1, 2)}
    u, v = sg["plays"].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert edge_set == {(0, 0), (1, 0), (1, 1)}
    assert F.array_equal(F.astype(inv["user"], idtype), F.tensor([0], idtype))

    # Test isolated node
    sg, inv = dgl.khop_out_subgraph(g, {"user": 3}, k=2)
    assert sg.idtype == idtype
    assert sg.num_nodes("game") == 0
    assert sg.num_nodes("user") == 1
    assert sg.num_edges("follows") == 0
    assert sg.num_edges("plays") == 0
    assert F.array_equal(F.astype(inv["user"], idtype), F.tensor([0], idtype))

    # Test multiple nodes
    sg, inv = dgl.khop_out_subgraph(
        g, {"user": F.tensor([2], idtype), "game": 0}, k=1
    )
    assert sg.num_edges("follows") == 0
    u, v = sg["plays"].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert edge_set == {(0, 1)}
    assert F.array_equal(F.astype(inv["user"], idtype), F.tensor([0], idtype))
    assert F.array_equal(F.astype(inv["game"], idtype), F.tensor([0], idtype))


@unittest.skipIf(not F.gpu_ctx(), "only necessary with GPU")
@pytest.mark.parametrize(
    "parent_idx_device",
    [("cpu", F.cpu()), ("cuda", F.cuda()), ("uva", F.cpu()), ("uva", F.cuda())],
)
@pytest.mark.parametrize("child_device", [F.cpu(), F.cuda()])
def test_subframes(parent_idx_device, child_device):
    parent_device, idx_device = parent_idx_device
    g = dgl.graph(
        (F.tensor([1, 2, 3], dtype=F.int64), F.tensor([2, 3, 4], dtype=F.int64))
    )
    print(g.device)
    g.ndata["x"] = F.randn((5, 4))
    g.edata["a"] = F.randn((3, 6))
    idx = F.tensor([1, 2], dtype=F.int64)
    if parent_device == "cuda":
        g = g.to(F.cuda())
    elif parent_device == "uva":
        if F.backend_name != "pytorch":
            pytest.skip("UVA only supported for PyTorch")
        g = g.to(F.cpu())
        g.create_formats_()
        g.pin_memory_()
    elif parent_device == "cpu":
        g = g.to(F.cpu())
    idx = F.copy_to(idx, idx_device)
    sg = g.sample_neighbors(idx, 2).to(child_device)
    assert sg.device == F.context(sg.ndata["x"])
    assert sg.device == F.context(sg.edata["a"])
    assert sg.device == child_device
    if parent_device != "uva":
        sg = g.to(child_device).sample_neighbors(
            F.copy_to(idx, child_device), 2
        )
        assert sg.device == F.context(sg.ndata["x"])
        assert sg.device == F.context(sg.edata["a"])
        assert sg.device == child_device
    if parent_device == "uva":
        g.unpin_memory_()


@unittest.skipIf(
    F._default_context_str != "gpu", reason="UVA only available on GPU"
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch",
    reason="UVA only supported for PyTorch",
)
@pytest.mark.parametrize("device", [F.cpu(), F.cuda()])
@parametrize_idtype
def test_uva_subgraph(idtype, device):
    g = create_test_heterograph2(idtype)
    g = g.to(F.cpu())
    g.create_formats_()
    g.pin_memory_()
    indices = {"user": F.copy_to(F.tensor([0], idtype), device)}
    edge_indices = {"follows": F.copy_to(F.tensor([0], idtype), device)}
    assert g.subgraph(indices).device == device
    assert g.edge_subgraph(edge_indices).device == device
    assert g.in_subgraph(indices).device == device
    assert g.out_subgraph(indices).device == device
    assert g.khop_in_subgraph(indices, 1)[0].device == device
    assert g.khop_out_subgraph(indices, 1)[0].device == device
    assert g.sample_neighbors(indices, 1).device == device
    g.unpin_memory_()


if __name__ == "__main__":
    test_edge_subgraph()
    test_uva_subgraph(F.int64, F.cpu())
    test_uva_subgraph(F.int64, F.cuda())
