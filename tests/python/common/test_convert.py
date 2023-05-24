import backend as F
import dgl


def get_nodes_by_ntype(nodes, ntype):
    return dict((k, v) for k, v in nodes.items() if v["label"] == ntype)


def get_edges_by_etype(edges, etype):
    return [e for e in edges if e[2]["triples"] == etype]


def check_attrs_for_nodes(nodes, attrs):
    return all(v.keys() == attrs for v in nodes.values())


def check_attr_values_for_nodes(nodes, attr_name, values):
    return F.allclose(
        F.stack([v[attr_name] for v in nodes.values()], 0), values
    )


def check_attrs_for_edges(edges, attrs):
    return all(e[2].keys() == attrs for e in edges)


def check_attr_values_for_edges(edges, attr_name, values):
    return F.allclose(F.stack([e[2][attr_name] for e in edges], 0), values)


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

    nxg = dgl.to_networkx(
        g,
        node_attrs=["n"],
        edge_attrs=["e"],
        ntype_attr="label",
        etype_attr="triples",
    )

    # Test nodes
    nxg_nodes = dict(nxg.nodes(data=True))
    assert len(nxg_nodes) == g.num_nodes()
    assert {v["label"] for v in nxg_nodes.values()} == set(g.ntypes)

    nxg_nodes_by_ntype = {}
    for ntype in g.ntypes:
        nxg_nodes_by_ntype[ntype] = get_nodes_by_ntype(nxg_nodes, ntype)
        assert g.num_nodes(ntype) == len(nxg_nodes_by_ntype[ntype])

    assert check_attrs_for_nodes(nxg_nodes_by_ntype["game"], {"label", "n"})
    assert check_attr_values_for_nodes(nxg_nodes_by_ntype["game"], "n", n1)
    assert check_attrs_for_nodes(nxg_nodes_by_ntype["user"], {"label", "n"})
    assert check_attr_values_for_nodes(nxg_nodes_by_ntype["user"], "n", n2)
    # Nodes without node attributes
    assert check_attrs_for_nodes(nxg_nodes_by_ntype["topic"], {"label"})

    # Test edges
    nxg_edges = list(nxg.edges(data=True))
    assert len(nxg_edges) == g.num_edges()
    assert {e[2]["triples"] for e in nxg_edges} == set(g.canonical_etypes)

    nxg_edges_by_etype = {}
    for etype in g.canonical_etypes:
        nxg_edges_by_etype[etype] = get_edges_by_etype(nxg_edges, etype)
        assert g.num_edges(etype) == len(nxg_edges_by_etype[etype])

    assert check_attrs_for_edges(
        nxg_edges_by_etype[("user", "follows", "user")],
        {"id", "triples", "e"},
    )
    assert check_attr_values_for_edges(
        nxg_edges_by_etype[("user", "follows", "user")], "e", e1
    )
    assert check_attrs_for_edges(
        nxg_edges_by_etype[("user", "plays", "game")], {"id", "triples", "e"}
    )
    assert check_attr_values_for_edges(
        nxg_edges_by_etype[("user", "plays", "game")], "e", e2
    )
    # Edges without edge attributes
    assert check_attrs_for_edges(
        nxg_edges_by_etype[("user", "follows", "topic")], {"id", "triples"}
    )
