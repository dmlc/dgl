import dgl
from dgl.contrib.cugraph.convert import cugraph_storage_from_heterograph
from dgl.contrib.cugraph import CuGraphStorage

import dgl.backend as F
import torch as th

device = "cuda"
ctx = th.device(device)


def create_test_heterograph1(idtype):
    graph_data = {
        ("nt.a", "join.1", "nt.a"): (
            F.tensor([0, 1, 2], dtype=idtype),
            F.tensor([0, 1, 2], dtype=idtype),
        ),
        ("nt.a", "join.2", "nt.a"): (
            F.tensor([0, 1, 2], dtype=idtype),
            F.tensor([0, 1, 2], dtype=idtype),
        ),
    }
    g = dgl.heterograph(graph_data, device=ctx)
    g.nodes["nt.a"].data["h"] = F.copy_to(
        F.tensor([1, 1, 1], dtype=idtype), ctx=ctx
    )
    return g


def create_test_heterograph2(idtype):
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
            ("developer", "tests", "game"): (
                F.tensor([0, 1], dtype=idtype),
                F.tensor([0, 1], dtype=idtype),
            ),
        },
        idtype=idtype,
        device=ctx,
    )

    g.nodes["user"].data["h"] = F.copy_to(
        F.tensor([1, 1, 1], dtype=idtype), ctx=ctx
    )
    g.nodes["user"].data["p"] = F.copy_to(
        F.tensor([1, 1, 1], dtype=idtype), ctx=ctx
    )
    g.nodes["game"].data["h"] = F.copy_to(
        F.tensor([2, 2], dtype=idtype), ctx=ctx
    )
    g.nodes["developer"].data["h"] = F.copy_to(
        F.tensor([3, 3], dtype=idtype), ctx=ctx
    )
    g.edges["plays"].data["h"] = F.copy_to(
        F.tensor([1, 1, 1, 1], dtype=idtype), ctx=ctx
    )
    return g


def create_test_heterograph3(idtype):
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
        device=ctx,
    )
    g.nodes["user"].data["h"] = F.copy_to(
        F.tensor([1, 1, 1], dtype=idtype), ctx=ctx
    )
    g.nodes["game"].data["h"] = F.copy_to(
        F.tensor([2, 2], dtype=idtype), ctx=ctx
    )
    g.edges["follows"].data["h"] = F.copy_to(
        F.tensor([10, 20, 30, 40, 50, 60], dtype=idtype), ctx=ctx
    )
    g.edges["follows"].data["p"] = F.copy_to(
        F.tensor([1, 2, 3, 4, 5, 6], dtype=idtype), ctx=ctx
    )
    g.edges["plays"].data["h"] = F.copy_to(
        F.tensor([1, 2], dtype=idtype), ctx=ctx
    )
    return g


def create_test_heterograph4(idtype):
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
        device=ctx,
    )
    g.nodes["user"].data["h"] = F.copy_to(
        F.tensor([1, 1, 1], dtype=idtype), ctx=ctx
    )
    g.nodes["game"].data["h"] = F.copy_to(
        F.tensor([2, 2], dtype=idtype), ctx=ctx
    )
    g.edges["follows"].data["h"] = F.copy_to(
        F.tensor([1, 2], dtype=idtype), ctx=ctx
    )
    g.edges["plays"].data["h"] = F.copy_to(
        F.tensor([1, 2], dtype=idtype), ctx=ctx
    )
    return g


def assert_same_num_nodes(gs, g):
    for ntype in g.ntypes:
        assert g.num_nodes(ntype) == gs.num_nodes(ntype)


def assert_same_num_edges_can_etypes(gs, g):
    for can_etype in g.canonical_etypes:
        assert g.num_edges(can_etype) == gs.num_edges(can_etype)


def assert_same_num_edges_etypes(gs, g):
    for etype in g.etypes:
        assert g.num_edges(etype) == gs.num_edges(etype)


def assert_same_edge_feats(gs, g):
    set(gs.graphstore.edata_feat_col_d.keys()) == set(g.edata.keys())
    for key in g.edata.keys():
        for etype in g.canonical_etypes:
            indices = th.arange(0, g.num_edges(etype), dtype=g.idtype).cuda()
            if len(g.etypes) <= 1 or etype in g.edata[key]:
                print(key, etype)
                g_output = g.get_edge_storage(key=key, etype=etype).fetch(
                    indices, device="cuda"
                )
                gs_output = gs.get_edge_storage(key=key, etype=etype).fetch(
                    indices
                )
                equal_t = (gs_output != g_output).sum().cpu()
                assert equal_t == 0


def assert_same_node_feats(gs, g):
    set(gs.graphstore.ndata_feat_col_d.keys()) == set(g.ndata.keys())

    for key in g.ndata.keys():
        for ntype in g.ntypes:
            indices = th.arange(0, g.num_nodes(ntype), dtype=g.idtype).cuda()
            if len(g.ntypes) <= 1 or ntype in g.ndata[key]:
                g_output = g.get_node_storage(key=key, ntype=ntype).fetch(
                    indices, device="cuda"
                )
                gs_output = gs.get_node_storage(key=key, ntype=ntype).fetch(
                    indices
                )
                equal_t = (gs_output != g_output).sum().cpu()
                assert equal_t == 0


def test_heterograph_conversion_nodes():
    graph_fs = [
        create_test_heterograph1,
        create_test_heterograph2,
        create_test_heterograph3,
        create_test_heterograph4,
    ]
    for graph_f in graph_fs:
        for idxtype in [th.int32, th.int64]:
            g = graph_f(idxtype)
            gs = cugraph_storage_from_heterograph(g)

            assert_same_num_nodes(gs, g)
            assert_same_node_feats(gs, g)


def test_heterograph_conversion_edges():
    graph_fs = [
        create_test_heterograph1,
        create_test_heterograph2,
        create_test_heterograph3,
        create_test_heterograph4,
    ]
    for graph_f in graph_fs:
        for idxtype in [th.int32, th.int64]:
            g = graph_f(idxtype)
            gs = cugraph_storage_from_heterograph(g)

            assert_same_num_edges_can_etypes(gs, g)
            assert_same_num_edges_etypes(gs, g)
            assert_same_edge_feats(gs, g)
