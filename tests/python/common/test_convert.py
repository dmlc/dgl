from collections import defaultdict as ddict

import dgl
import networkx as nx

import backend as F


def _test_nx_conversion():
    # check conversion between networkx and DGLGraph

    def _check_nx_feature(nxg, nf, ef):
        # check node and edge feature of nxg
        # this is used to check to_networkx
        num_nodes = len(nxg)
        num_edges = nxg.size()
        if num_nodes > 0:
            node_feat = ddict(list)
            for nid, attr in nxg.nodes(data=True):
                assert len(attr) == len(nf)
                for k in nxg.nodes[nid]:
                    node_feat[k].append(F.unsqueeze(attr[k], 0))
            for k in node_feat:
                feat = F.cat(node_feat[k], 0)
                assert F.allclose(feat, nf[k])
        else:
            assert len(nf) == 0
        if num_edges > 0:
            edge_feat = ddict(lambda: [0] * num_edges)
            for u, v, attr in nxg.edges(data=True):
                assert len(attr) == len(ef) + 1  # extra id
                eid = attr["id"]
                for k in ef:
                    edge_feat[k][eid] = F.unsqueeze(attr[k], 0)
            for k in edge_feat:
                feat = F.cat(edge_feat[k], 0)
                assert F.allclose(feat, ef[k])
        else:
            assert len(ef) == 0

    n1 = F.randn((5, 3))
    n2 = F.randn((5, 10))
    n3 = F.randn((5, 4))
    e1 = F.randn((4, 5))
    e2 = F.randn((4, 7))
    g = dgl.graph(([0, 1, 3, 4], [2, 4, 0, 3]))
    g.ndata.update({"n1": n1, "n2": n2, "n3": n3})
    g.edata.update({"e1": e1, "e2": e2})

    # convert to networkx
    nxg = g.to_networkx(node_attrs=["n1", "n3"], edge_attrs=["e1", "e2"])
    assert len(nxg) == 5
    assert nxg.size() == 4
    _check_nx_feature(nxg, {"n1": n1, "n3": n3}, {"e1": e1, "e2": e2})

    # convert to DGLGraph, nx graph has id in edge feature
    # use id feature to test non-tensor copy
    g = dgl.from_networkx(nxg, node_attrs=["n1"], edge_attrs=["e1", "id"])
    # check graph size
    assert g.num_nodes() == 5
    assert g.num_edges() == 4
    # check number of features
    # test with existing dglgraph (so existing features should be cleared)
    assert len(g.ndata) == 1
    assert len(g.edata) == 2
    # check feature values
    assert F.allclose(g.ndata["n1"], n1)
    # with id in nx edge feature, e1 should follow original order
    assert F.allclose(g.edata["e1"], e1)
    assert F.array_equal(F.astype(g.edata["id"], F.int64), F.copy_to(F.arange(0, 4), F.cpu()))

    # test conversion after modifying DGLGraph
    g.edata.pop("id")  # pop id so we don't need to provide id when adding edges
    new_n = F.randn((2, 3))
    new_e = F.randn((3, 5))
    g.add_nodes(2, data={"n1": new_n})
    # add three edges, one is a multi-edge
    g.add_edges([3, 6, 0], [4, 5, 2], data={"e1": new_e})
    n1 = F.cat((n1, new_n), 0)
    e1 = F.cat((e1, new_e), 0)
    # convert to networkx again
    nxg = g.to_networkx(node_attrs=["n1"], edge_attrs=["e1"])
    assert len(nxg) == 7
    assert nxg.size() == 7
    _check_nx_feature(nxg, {"n1": n1}, {"e1": e1})

    # now test convert from networkx without id in edge feature
    # first pop id in edge feature
    for _, _, attr in nxg.edges(data=True):
        attr.pop("id")
    # test with a new graph
    g = dgl.from_networkx(nxg, node_attrs=["n1"], edge_attrs=["e1"])
    # check graph size
    assert g.num_nodes() == 7
    assert g.num_edges() == 7
    # check number of features
    assert len(g.ndata) == 1
    assert len(g.edata) == 1
    # check feature values
    assert F.allclose(g.ndata["n1"], n1)
    # edge feature order follows nxg.edges()
    edge_feat = []
    for _, _, attr in nxg.edges(data=True):
        edge_feat.append(F.unsqueeze(attr["e1"], 0))
    edge_feat = F.cat(edge_feat, 0)
    assert F.allclose(g.edata["e1"], edge_feat)

    # Test converting from a networkx graph whose nodes are
    # not labeled with consecutive-integers.
    nxg = nx.cycle_graph(5)
    nxg.remove_nodes_from([0, 4])
    for u in nxg.nodes():
        nxg.nodes[u]["h"] = F.tensor([u])
    for u, v, d in nxg.edges(data=True):
        d["h"] = F.tensor([u, v])

    g = dgl.from_networkx(nxg, node_attrs=["h"], edge_attrs=["h"])
    assert g.num_nodes() == 3
    assert g.num_edges() == 4
    assert g.has_edge_between(0, 1)
    assert g.has_edge_between(1, 2)
    assert F.allclose(g.ndata["h"], F.tensor([[1.0], [2.0], [3.0]]))
    assert F.allclose(g.edata["h"], F.tensor([[1.0, 2.0], [1.0, 2.0], [2.0, 3.0], [2.0, 3.0]]))


def test_nx_convert_heterogeneous():
    g = dgl.heterograph(
        {
            ("user", "follows", "user"): ([0, 1], [1, 2]),
            ("user", "follows", "topic"): ([1, 1], [1, 2]),
            ("user", "plays", "game"): ([0, 3], [3, 4]),
        }
    )

    n1 = F.randn((5, 3))
    n2 = F.randn((4, 2))
    e1 = F.randn((2, 3))
    e2 = F.randn((2, 2))

    g.ndata["n"] = {"game": n1, "user": n2}
    g.edata["e"] = {("user", "follows", "user"): e1, "plays": e2}

    nxg = dgl.to_networkx(g, node_attrs=["n"], edge_attrs=["e"])

    # Test nodes
    nxg_nodes = dict(nxg.nodes(data=True))
    assert len(nxg_nodes) == g.num_nodes()
    assert {v["label"] for v in nxg_nodes.values()} == set(g.ntypes)

    nxg_nodes_by_label = {}
    for ntype in g.ntypes:
        nxg_nodes_by_label[ntype] = dict((k, v) for k, v in nxg_nodes.items() if v["label"] == ntype)
        assert g.num_nodes(ntype) == len(nxg_nodes_by_label[ntype])

    assert all(v.keys() == {"label", "n"} for v in nxg_nodes_by_label["game"].values())
    assert F.allclose(F.stack([v["n"] for v in nxg_nodes_by_label["game"].values()], 0), n1)
    assert all(v.keys() == {"label", "n"} for v in nxg_nodes_by_label["user"].values())
    assert F.allclose(F.stack([v["n"] for v in nxg_nodes_by_label["user"].values()], 0), n2)
    # Nodes without node attributes
    assert all(v.keys() == {"label"} for v in nxg_nodes_by_label["topic"].values())

    # Test edges
    nxg_edges = list(nxg.edges(data=True))
    assert len(nxg_edges) == g.num_edges()
    assert {e[2]["triples"] for e in nxg_edges} == set(g.canonical_etypes)

    nxg_edges_by_triples = {}
    for etype in g.canonical_etypes:
        nxg_edges_by_triples[etype] = sorted(
            [e for e in nxg_edges if e[2]["triples"] == etype], key=lambda el: el[2]["id"]
        )
        assert g.num_edges(etype) == len(nxg_edges_by_triples[etype])

    assert all(e[2].keys() == {"id", "triples", "e"} for e in nxg_edges_by_triples[("user", "follows", "user")])
    assert F.allclose(F.stack([e[2]["e"] for e in nxg_edges_by_triples[("user", "follows", "user")]], 0), e1)
    assert all(e[2].keys() == {"id", "triples", "e"} for e in nxg_edges_by_triples[("user", "plays", "game")])
    assert F.allclose(F.stack([e[2]["e"] for e in nxg_edges_by_triples[("user", "plays", "game")]], 0), e2)
    # Edges without edge attributes
    assert all(e[2].keys() == {"id", "triples"} for e in nxg_edges_by_triples[("user", "follows", "topic")])
