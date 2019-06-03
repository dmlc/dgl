import os
import backend as F
import networkx as nx
import numpy as np
import torch as th
import dgl

def test_node_removal():
    g = dgl.DGLGraph()
    g.add_nodes(10)
    assert g.number_of_nodes() == 10
    g.ndata['id'] = F.arange(10)

    # delete nodes
    g.del_nodes(range(4, 7))
    assert g.number_of_nodes() == 7
    assert F.array_equal(g.ndata['id'], F.tensor([0, 1, 2, 3, 7, 8, 9]))

    # add nodes
    g.add_nodes(3)
    assert g.number_of_nodes() == 10
    assert F.array_equal(g.ndata['id'], F.tensor([0, 1, 2, 3, 7, 8, 9, 0, 0, 0]))

    # delete nodes
    g.del_nodes(range(1, 4))
    assert g.number_of_nodes() == 7
    assert F.array_equal(g.ndata['id'], F.tensor([0, 7, 8, 9, 0, 0, 0]))

def test_edge_removal():
    g = dgl.DGLGraph()
    g.add_nodes(5)
    for i in range(5):
        for j in range(5):
            g.add_edge(i, j)
    g.edata['id'] = F.arange(25)

    # delete edges
    g.del_edges(range(13, 20))
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 18
    assert F.array_equal(g.edata['id'], F.tensor(list(range(13)) + list(range(20, 25))))

    # add edges
    g.add_edge(3, 3)
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 19
    assert F.array_equal(g.edata['id'], F.tensor(list(range(13)) + list(range(20, 25)) + [0]))

    # delete edges
    g.del_edges(range(2, 10))
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 11
    assert F.array_equal(g.edata['id'], F.tensor([0, 1, 10, 11, 12, 20, 21, 22, 23, 24, 0]))

def test_node_and_edge_removal():
    g = dgl.DGLGraph()
    g.add_nodes(10)
    for i in range(10):
        for j in range(10):
            g.add_edge(i, j)
    g.edata['id'] = F.arange(100)
    assert g.number_of_nodes() == 10
    assert g.number_of_edges() == 100

    # delete nodes
    g.del_nodes([2, 4])
    assert g.number_of_nodes() == 8
    assert g.number_of_edges() == 64

    # delete edges
    g.del_edges(range(10, 20))
    assert g.number_of_nodes() == 8
    assert g.number_of_edges() == 54

    # add nodes
    g.add_nodes(2)
    assert g.number_of_nodes() == 10
    assert g.number_of_edges() == 54

    # add edges
    for i in range(8, 10):
        for j in range(8, 10):
            g.add_edge(i, j)
    assert g.number_of_nodes() == 10
    assert g.number_of_edges() == 58

    # delete edges
    g.del_edges(range(10, 20))
    assert g.number_of_nodes() == 10
    assert g.number_of_edges() == 48

if __name__ == '__main__':
    test_node_removal()
    test_edge_removal()
    test_node_and_edge_removal()
