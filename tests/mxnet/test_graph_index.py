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

def check_basics(g, ig):
    assert g.number_of_nodes() == ig.number_of_nodes()
    assert g.number_of_edges() == ig.number_of_edges()

    edges = g.edges()
    iedges = ig.edges()

    for i in range(g.number_of_nodes()):
        assert g.has_node(i) == ig.has_node(i)

    for i in range(g.number_of_nodes()):
        assert mx.nd.sum(g.predecessors(i).tousertensor()).asnumpy() == mx.nd.sum(ig.predecessors(i).tousertensor()).asnumpy()
        assert mx.nd.sum(g.successors(i).tousertensor()).asnumpy() == mx.nd.sum(ig.successors(i).tousertensor()).asnumpy()

    randv = np.random.randint(0, g.number_of_nodes(), 10)
    randv = utils.toindex(randv)
    in_src1, in_dst1, in_eids1 = g.in_edges(randv)
    in_src2, in_dst2, in_eids2 = ig.in_edges(randv)
    nnz = in_src2.tousertensor().shape[0]
    assert mx.nd.sum(in_src1.tousertensor() == in_src2.tousertensor()).asnumpy() == nnz
    assert mx.nd.sum(in_dst1.tousertensor() == in_dst2.tousertensor()).asnumpy() == nnz
    assert mx.nd.sum(in_eids1.tousertensor() == in_eids2.tousertensor()).asnumpy() == nnz

    out_src1, out_dst1, out_eids1 = g.out_edges(randv)
    out_src2, out_dst2, out_eids2 = ig.out_edges(randv)
    nnz = out_dst2.tousertensor().shape[0]
    assert mx.nd.sum(out_dst1.tousertensor() == out_dst2.tousertensor()).asnumpy() == nnz
    assert mx.nd.sum(out_src1.tousertensor() == out_src2.tousertensor()).asnumpy() == nnz
    assert mx.nd.sum(out_eids1.tousertensor() == out_eids2.tousertensor()).asnumpy() == nnz

def test_basics():
    g, ig = generate_rand_graph(100)
    check_basics(g, ig)

def test_node_subgraph():
    num_vertices = 100
    g, ig = generate_rand_graph(num_vertices)

    # node_subgraph
    randv1 = np.random.randint(0, num_vertices, 20)
    randv = np.unique(randv1)
    subg = g.node_subgraph(utils.toindex(randv))
    subig = ig.node_subgraph(utils.toindex(randv))
    check_graph_equal(subg, subig)
    assert mx.nd.sum(map_to_subgraph_nid(subg, randv1[0:10]).tousertensor()
            == map_to_subgraph_nid(subig, randv1[0:10]).tousertensor()) == 10

    # node_subgraphs
    randvs = []
    subgs = []
    for i in range(4):
        randv = np.unique(np.random.randint(0, num_vertices, 20))
        randvs.append(utils.toindex(randv))
        subgs.append(g.node_subgraph(utils.toindex(randv)))
    subigs= ig.node_subgraphs(randvs)
    for i in range(4):
        check_graph_equal(subgs[i], subigs[i])


test_basics()
test_graph_gen()
test_node_subgraph()
