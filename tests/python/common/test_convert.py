import unittest

import backend as F
import dgl

from utils import parametrize_idtype


def get_nodes_by_ntype(nodes, ntype):
    return dict((k, v) for k, v in nodes.items() if v["ntype"] == ntype)


def edge_attrs(edge):
    # Edges in Networkx are in the format (src, dst, attrs)
    return edge[2]


def get_edges_by_etype(edges, etype):
    return [e for e in edges if edge_attrs(e)["etype"] == etype]


def check_attrs_for_nodes(nodes, attrs):
    return all(v.keys() == attrs for v in nodes.values())


def check_attr_values_for_nodes(nodes, attr_name, values):
    return F.allclose(
        F.stack([v[attr_name] for v in nodes.values()], 0), values
    )


def check_attrs_for_edges(edges, attrs):
    return all(edge_attrs(e).keys() == attrs for e in edges)


def check_attr_values_for_edges(edges, attr_name, values):
    return F.allclose(
        F.stack([edge_attrs(e)[attr_name] for e in edges], 0), values
    )


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="`to_networkx` does not support graphs on GPU",
)
@parametrize_idtype
def test_to_networkx(idtype):
    # TODO: adapt and move code from the _test_nx_conversion function in
    # tests/python/common/function/test_basics.py to here
    # (pending resolution of https://github.com/dmlc/dgl/issues/5735).
    g = dgl.heterograph(
        {
            ("user", "follows", "user"): ([0, 1], [1, 2]),
            ("user", "follows", "topic"): ([1, 1], [1, 2]),
            ("user", "plays", "game"): ([0, 3], [3, 4]),
        },
        idtype=idtype,
        device=F.ctx(),
    )

    n1 = F.randn((5, 3))
    n2 = F.randn((4, 2))
    e1 = F.randn((2, 3))
    e2 = F.randn((2, 2))

    g.nodes["game"].data["n"] = F.copy_to(n1, ctx=F.ctx())
    g.nodes["user"].data["n"] = F.copy_to(n2, ctx=F.ctx())
    g.edges[("user", "follows", "user")].data["e"] = F.copy_to(e1, ctx=F.ctx())
    g.edges["plays"].data["e"] = F.copy_to(e2, ctx=F.ctx())

    nxg = dgl.to_networkx(
        g,
        node_attrs=["n"],
        edge_attrs=["e"],
    )

    # Test nodes
    nxg_nodes = dict(nxg.nodes(data=True))
    assert len(nxg_nodes) == g.num_nodes()
    assert {v["ntype"] for v in nxg_nodes.values()} == set(g.ntypes)

    nxg_nodes_by_ntype = {}
    for ntype in g.ntypes:
        nxg_nodes_by_ntype[ntype] = get_nodes_by_ntype(nxg_nodes, ntype)
        assert g.num_nodes(ntype) == len(nxg_nodes_by_ntype[ntype])

    assert check_attrs_for_nodes(nxg_nodes_by_ntype["game"], {"ntype", "n"})
    assert check_attr_values_for_nodes(nxg_nodes_by_ntype["game"], "n", n1)
    assert check_attrs_for_nodes(nxg_nodes_by_ntype["user"], {"ntype", "n"})
    assert check_attr_values_for_nodes(nxg_nodes_by_ntype["user"], "n", n2)
    # Nodes without node attributes
    assert check_attrs_for_nodes(nxg_nodes_by_ntype["topic"], {"ntype"})

    # Test edges
    nxg_edges = list(nxg.edges(data=True))
    assert len(nxg_edges) == g.num_edges()
    assert {edge_attrs(e)["etype"] for e in nxg_edges} == set(
        g.canonical_etypes
    )

    nxg_edges_by_etype = {}
    for etype in g.canonical_etypes:
        nxg_edges_by_etype[etype] = get_edges_by_etype(nxg_edges, etype)
        assert g.num_edges(etype) == len(nxg_edges_by_etype[etype])

    assert check_attrs_for_edges(
        nxg_edges_by_etype[("user", "follows", "user")],
        {"id", "etype", "e"},
    )
    assert check_attr_values_for_edges(
        nxg_edges_by_etype[("user", "follows", "user")], "e", e1
    )
    assert check_attrs_for_edges(
        nxg_edges_by_etype[("user", "plays", "game")], {"id", "etype", "e"}
    )
    assert check_attr_values_for_edges(
        nxg_edges_by_etype[("user", "plays", "game")], "e", e2
    )
    # Edges without edge attributes
    assert check_attrs_for_edges(
        nxg_edges_by_etype[("user", "follows", "topic")], {"id", "etype"}
    )
