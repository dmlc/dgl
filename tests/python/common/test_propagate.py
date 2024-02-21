import unittest

import backend as F

import dgl
import networkx as nx
from utils import check_fail, parametrize_idtype


def create_graph(idtype):
    g = dgl.from_networkx(nx.path_graph(5), idtype=idtype, device=F.ctx())
    return g


def mfunc(edges):
    return {"m": edges.src["x"]}


def rfunc(nodes):
    msg = F.sum(nodes.mailbox["m"], 1)
    return {"x": nodes.data["x"] + msg}


@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
@parametrize_idtype
def test_prop_nodes_bfs(idtype):
    g = create_graph(idtype)
    g.ndata["x"] = F.ones((5, 2))
    dgl.prop_nodes_bfs(
        g, 0, message_func=mfunc, reduce_func=rfunc, apply_node_func=None
    )
    # pull nodes using bfs order will result in a cumsum[i] + data[i] + data[i+1]
    assert F.allclose(
        g.ndata["x"],
        F.tensor([[2.0, 2.0], [4.0, 4.0], [6.0, 6.0], [8.0, 8.0], [9.0, 9.0]]),
    )


@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
@parametrize_idtype
def test_prop_edges_dfs(idtype):
    g = create_graph(idtype)
    g.ndata["x"] = F.ones((5, 2))
    dgl.prop_edges_dfs(
        g, 0, message_func=mfunc, reduce_func=rfunc, apply_node_func=None
    )
    # snr using dfs results in a cumsum
    assert F.allclose(
        g.ndata["x"],
        F.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]),
    )

    g.ndata["x"] = F.ones((5, 2))
    dgl.prop_edges_dfs(
        g,
        0,
        has_reverse_edge=True,
        message_func=mfunc,
        reduce_func=rfunc,
        apply_node_func=None,
    )
    # result is cumsum[i] + cumsum[i-1]
    assert F.allclose(
        g.ndata["x"],
        F.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [7.0, 7.0], [9.0, 9.0]]),
    )

    g.ndata["x"] = F.ones((5, 2))
    dgl.prop_edges_dfs(
        g,
        0,
        has_nontree_edge=True,
        message_func=mfunc,
        reduce_func=rfunc,
        apply_node_func=None,
    )
    # result is cumsum[i] + cumsum[i+1]
    assert F.allclose(
        g.ndata["x"],
        F.tensor([[3.0, 3.0], [5.0, 5.0], [7.0, 7.0], [9.0, 9.0], [5.0, 5.0]]),
    )


@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
@parametrize_idtype
def test_prop_nodes_topo(idtype):
    # bi-directional chain
    g = create_graph(idtype)
    assert check_fail(dgl.prop_nodes_topo, g)  # has loop

    # tree
    tree = dgl.graph([])
    tree.add_nodes(5)
    tree.add_edges(1, 0)
    tree.add_edges(2, 0)
    tree.add_edges(3, 2)
    tree.add_edges(4, 2)
    tree = dgl.graph(tree.edges())
    # init node feature data
    tree.ndata["x"] = F.zeros((5, 2))
    # set all leaf nodes to be ones
    tree.nodes[[1, 3, 4]].data["x"] = F.ones((3, 2))

    # Filtering DGLWarning:
    #    The input graph for the user-defined edge
    #    function does not contain valid edges
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        dgl.prop_nodes_topo(
            tree, message_func=mfunc, reduce_func=rfunc, apply_node_func=None
        )
    # root node get the sum
    assert F.allclose(tree.nodes[0].data["x"], F.tensor([[3.0, 3.0]]))


if __name__ == "__main__":
    test_prop_nodes_bfs()
    test_prop_edges_dfs()
    test_prop_nodes_topo()
