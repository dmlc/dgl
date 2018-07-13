import networkx as nx
import numpy as np
from dgl.graph import DGLGraph

def test_node1():
    graph = DGLGraph()
    n0 = 0
    n1 = 1
    graph.add_node(n0, x=10)
    graph.add_node(n1, x=11)
    assert len(graph.nodes()) == 2
    assert graph.node[[n0, n1]]['x'] == [10, 11]
    # tensor state
    graph.add_node(n0, y=np.zeros((1, 10)))
    graph.add_node(n1, y=np.zeros((1, 10)))
    assert graph.node[[n0, n1]]['y'].shape == (2, 10)
    # tensor args
#   nodes = np.array([n0, n1, n1, n0])
#   assert graph.node[nodes]['y'].shape == (4, 10)

def test_node2():
    g = DGLGraph()
    n0 = 0
    n1 = 1
    g.add_node([n0, n1])
    assert len(g.nodes()) == 2

def test_edge1():
    g = DGLGraph()
    g.add_node(list(range(10)))  # add 10 nodes.
    g.add_edge(0, 1, x=10)
    assert g.number_of_edges() == 1
    assert g[0][1]['x'] == 10
    # add many-many edges
    u = [1, 2, 3]
    v = [2, 3, 4]
    g.add_edge(u, v, y=11)  # add 3 edges.
    assert g.number_of_edges() == 4
    assert g[u][v]['y'] == [11, 11, 11]
    # add one-many edges
    u = 5
    v = [6, 7]
    g.add_edge(u, v, y=22)  # add 2 edges.
    assert g.number_of_edges() == 6
    assert g[u][v]['y'] == [22, 22]
    # add many-one edges
    u = [8, 9]
    v = 7
    g.add_edge(u, v, y=33)  # add 2 edges.
    assert g.number_of_edges() == 8
    assert g[u][v]['y'] == [33, 33]
    # tensor type edge attr
    z = np.zeros((5, 10))  # 5 edges, each of is (10,) vector
    u = [1, 2, 3, 5, 8]
    v = [2, 3, 4, 6, 7]
    g[u][v]['z'] = z
    u = np.array(u)
    v = np.array(v)
    assert g[u][v]['z'].shape == (5, 10)

def test_graph1():
    g = DGLGraph(nx.path_graph(3))

def test_view():
    g = DGLGraph(nx.path_graph(3))
    g.nodes[0]
    g.edges[0, 1]
    u = [0, 1]
    v = [1, 2]
    g.nodes[u]
    g.edges[u, v]['x'] = 1
    assert g.edges[u, v]['x'] == [1, 1]

test_node1()
test_node2()
test_edge1()
test_graph1()
test_view()
