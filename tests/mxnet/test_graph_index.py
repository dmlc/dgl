import os
os.environ['DGLBACKEND'] = 'mxnet'
import mxnet as mx
import numpy as np
import scipy as sp
from dgl.graph import GraphIndex, create_graph_index
from dgl.graph_index import map_to_subgraph_nid
import dgl.backend as F
from dgl import utils

def generate_graph():
    g = create_graph_index()
    g.add_nodes(10) # 10 nodes.
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    # add a back flow from 9 to 0
    g.add_edge(9, 0)
    ig = create_graph_index(g.to_networkx(), immutable_graph=True)
    return g, ig

def generate_rand_graph(n):
    arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
    g = create_graph_index(arr)
    ig = create_graph_index(arr, immutable_graph=True)
    return g, ig

def check_graph_equal(g1, g2):
    ctx = F.get_context(mx.nd.array([1]))
    adj1 = g1.adjacency_matrix().get(ctx) != 0
    adj2 = g2.adjacency_matrix().get(ctx) != 0
    assert mx.nd.sum(adj1 - adj2).asnumpy() == 0

def test_graph_gen():
    g, ig = generate_rand_graph(10)
    check_graph_equal(g, ig)

def test_basics():
    g, ig = generate_rand_graph(100)
    assert g.number_of_nodes() == ig.number_of_nodes()
    assert g.number_of_edges() == ig.number_of_edges()

    edges = g.edges()
    iedges = ig.edges()

    for i in range(g.number_of_nodes()):
        assert g.has_node(i) == ig.has_node(i)

    for i in range(g.number_of_nodes()):
        assert mx.nd.sum(g.predecessors(i).tousertensor()).asnumpy() == mx.nd.sum(ig.predecessors(i).tousertensor()).asnumpy()
        assert mx.nd.sum(g.successors(i).tousertensor()).asnumpy() == mx.nd.sum(ig.successors(i).tousertensor()).asnumpy()

def test_node_subgraph():
    num_vertices = 100
    g, ig = generate_rand_graph(num_vertices)
    randv1 = np.random.randint(0, num_vertices, 20)
    randv = np.unique(randv1)
    subg = g.node_subgraph(utils.toindex(randv))
    subig = ig.node_subgraph(utils.toindex(randv))
    check_graph_equal(subg, subig)
    assert mx.nd.sum(map_to_subgraph_nid(subg, randv1[0:10]).tousertensor()
            == map_to_subgraph_nid(subig, randv1[0:10]).tousertensor()) == 10

test_basics()
test_graph_gen()
test_node_subgraph()
