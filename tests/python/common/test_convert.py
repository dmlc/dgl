import dgl

import backend as F


def test_to_networkx():
    # TODO: adapt and move code from the _test_nx_conversion function in
    # tests/python/common/function/test_basics.py to here
    # (pending resolution of https://github.com/dmlc/dgl/issues/5735).
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
