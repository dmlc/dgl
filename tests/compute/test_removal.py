import backend as F
import numpy as np
from test_utils import parametrize_idtype

import dgl


@parametrize_idtype
def test_node_removal(idtype):
    g = dgl.DGLGraph()
    g = g.astype(idtype).to(F.ctx())
    g.add_nodes(10)
    g.add_edges(0, 0)
    assert g.number_of_nodes() == 10
    g.ndata["id"] = F.arange(0, 10)

    # remove nodes
    g.remove_nodes(range(4, 7))
    assert g.number_of_nodes() == 7
    assert F.array_equal(g.ndata["id"], F.tensor([0, 1, 2, 3, 7, 8, 9]))
    assert dgl.NID not in g.ndata
    assert dgl.EID not in g.edata

    # add nodes
    g.add_nodes(3)
    assert g.number_of_nodes() == 10
    assert F.array_equal(
        g.ndata["id"], F.tensor([0, 1, 2, 3, 7, 8, 9, 0, 0, 0])
    )

    # remove nodes
    g.remove_nodes(range(1, 4), store_ids=True)
    assert g.number_of_nodes() == 7
    assert F.array_equal(g.ndata["id"], F.tensor([0, 7, 8, 9, 0, 0, 0]))
    assert dgl.NID in g.ndata
    assert dgl.EID in g.edata


@parametrize_idtype
def test_multigraph_node_removal(idtype):
    g = dgl.DGLGraph()
    g = g.astype(idtype).to(F.ctx())
    g.add_nodes(5)
    for i in range(5):
        g.add_edges(i, i)
        g.add_edges(i, i)
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 10

    # remove nodes
    g.remove_nodes([2, 3])
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 6

    # add nodes
    g.add_nodes(1)
    g.add_edges(1, 1)
    g.add_edges(1, 1)
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() == 8

    # remove nodes
    g.remove_nodes([0])
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 6


@parametrize_idtype
def test_multigraph_edge_removal(idtype):
    g = dgl.DGLGraph()
    g = g.astype(idtype).to(F.ctx())
    g.add_nodes(5)
    for i in range(5):
        g.add_edges(i, i)
        g.add_edges(i, i)
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 10

    # remove edges
    g.remove_edges([2, 3])
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 8

    # add edges
    g.add_edges(1, 1)
    g.add_edges(1, 1)
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 10

    # remove edges
    g.remove_edges([0, 1])
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 8


@parametrize_idtype
def test_edge_removal(idtype):
    g = dgl.DGLGraph()
    g = g.astype(idtype).to(F.ctx())
    g.add_nodes(5)
    for i in range(5):
        for j in range(5):
            g.add_edges(i, j)
    g.edata["id"] = F.arange(0, 25)

    # remove edges
    g.remove_edges(range(13, 20))
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 18
    assert F.array_equal(
        g.edata["id"], F.tensor(list(range(13)) + list(range(20, 25)))
    )
    assert dgl.NID not in g.ndata
    assert dgl.EID not in g.edata

    # add edges
    g.add_edges(3, 3)
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 19
    assert F.array_equal(
        g.edata["id"], F.tensor(list(range(13)) + list(range(20, 25)) + [0])
    )

    # remove edges
    g.remove_edges(range(2, 10), store_ids=True)
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 11
    assert F.array_equal(
        g.edata["id"], F.tensor([0, 1, 10, 11, 12, 20, 21, 22, 23, 24, 0])
    )
    assert dgl.EID in g.edata


@parametrize_idtype
def test_node_and_edge_removal(idtype):
    g = dgl.DGLGraph()
    g = g.astype(idtype).to(F.ctx())
    g.add_nodes(10)
    for i in range(10):
        for j in range(10):
            g.add_edges(i, j)
    g.edata["id"] = F.arange(0, 100)
    assert g.number_of_nodes() == 10
    assert g.number_of_edges() == 100

    # remove nodes
    g.remove_nodes([2, 4])
    assert g.number_of_nodes() == 8
    assert g.number_of_edges() == 64

    # remove edges
    g.remove_edges(range(10, 20))
    assert g.number_of_nodes() == 8
    assert g.number_of_edges() == 54

    # add nodes
    g.add_nodes(2)
    assert g.number_of_nodes() == 10
    assert g.number_of_edges() == 54

    # add edges
    for i in range(8, 10):
        for j in range(8, 10):
            g.add_edges(i, j)
    assert g.number_of_nodes() == 10
    assert g.number_of_edges() == 58

    # remove edges
    g.remove_edges(range(10, 20))
    assert g.number_of_nodes() == 10
    assert g.number_of_edges() == 48


@parametrize_idtype
def test_node_frame(idtype):
    g = dgl.DGLGraph()
    g = g.astype(idtype).to(F.ctx())
    g.add_nodes(10)
    data = np.random.rand(10, 3)
    new_data = data.take([0, 1, 2, 7, 8, 9], axis=0)
    g.ndata["h"] = F.tensor(data)

    # remove nodes
    g.remove_nodes(range(3, 7))
    assert F.allclose(g.ndata["h"], F.tensor(new_data))


@parametrize_idtype
def test_edge_frame(idtype):
    g = dgl.DGLGraph()
    g = g.astype(idtype).to(F.ctx())
    g.add_nodes(10)
    g.add_edges(list(range(10)), list(range(1, 10)) + [0])
    data = np.random.rand(10, 3)
    new_data = data.take([0, 1, 2, 7, 8, 9], axis=0)
    g.edata["h"] = F.tensor(data)

    # remove edges
    g.remove_edges(range(3, 7))
    assert F.allclose(g.edata["h"], F.tensor(new_data))


@parametrize_idtype
def test_issue1287(idtype):
    # reproduce https://github.com/dmlc/dgl/issues/1287.
    # setting features after remove nodes
    g = dgl.DGLGraph()
    g = g.astype(idtype).to(F.ctx())
    g.add_nodes(5)
    g.add_edges([0, 2, 3, 1, 1], [1, 0, 3, 1, 0])
    g.remove_nodes([0, 1])
    g.ndata["h"] = F.randn((g.number_of_nodes(), 3))
    g.edata["h"] = F.randn((g.number_of_edges(), 2))

    # remove edges
    g = dgl.DGLGraph()
    g = g.astype(idtype).to(F.ctx())
    g.add_nodes(5)
    g.add_edges([0, 2, 3, 1, 1], [1, 0, 3, 1, 0])
    g.remove_edges([0, 1])
    g = g.to(F.ctx())
    g.ndata["h"] = F.randn((g.number_of_nodes(), 3))
    g.edata["h"] = F.randn((g.number_of_edges(), 2))


if __name__ == "__main__":
    test_node_removal()
    test_edge_removal()
    test_multigraph_node_removal()
    test_multigraph_edge_removal()
    test_node_and_edge_removal()
    test_node_frame()
    test_edge_frame()
    test_frame_size()
