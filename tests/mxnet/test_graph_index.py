import os
os.environ['DGLBACKEND'] = 'mxnet'
import mxnet as mx
import numpy as np
from dgl.graph import GraphIndex, create_graph_index
from dgl import create_immutable_graph_index

def generate_graph():
    g = create_graph_index()
    g.add_nodes(10) # 10 nodes.
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    # add a back flow from 9 to 0
    g.add_edge(9, 0)
    ig = create_immutable_graph_index(g.to_networkx())
    return g, ig

def test_basics():
    g, ig = generate_graph()
    assert g.number_of_nodes() == ig.number_of_nodes()
    assert g.number_of_edges() == ig.number_of_edges()

    for i in range(g.number_of_nodes()):
        assert g.has_node(i) == ig.has_node(i)

    for i in range(g.number_of_nodes()):
        assert mx.nd.sum(g.predecessors(i).tousertensor()).asnumpy() == mx.nd.sum(ig.predecessors(i).tousertensor()).asnumpy()
        assert mx.nd.sum(g.successors(i).tousertensor()).asnumpy() == mx.nd.sum(ig.successors(i).tousertensor()).asnumpy()

test_basics()
