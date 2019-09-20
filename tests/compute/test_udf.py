import backend as F
import dgl
import networkx as nx
import dgl.utils as utils
from dgl import DGLGraph, ALL
from dgl.udf import NodeBatch, EdgeBatch

def test_node_batch():
    g = dgl.DGLGraph(nx.path_graph(20))
    feat = F.randn((g.number_of_nodes(), 10))
    g.ndata['x'] = feat

    # test all
    v = utils.toindex(slice(0, g.number_of_nodes()))
    n_repr = g.get_n_repr(v)
    nbatch = NodeBatch(v, n_repr)
    assert F.allclose(nbatch.data['x'], feat)
    assert nbatch.mailbox is None
    assert F.allclose(nbatch.nodes(), g.nodes())
    assert nbatch.batch_size() == g.number_of_nodes()
    assert len(nbatch) == g.number_of_nodes()

    # test partial
    v = utils.toindex(F.tensor([0, 3, 5, 7, 9]))
    n_repr = g.get_n_repr(v)
    nbatch = NodeBatch(v, n_repr)
    assert F.allclose(nbatch.data['x'], F.gather_row(feat, F.tensor([0, 3, 5, 7, 9])))
    assert nbatch.mailbox is None
    assert F.allclose(nbatch.nodes(), F.tensor([0, 3, 5, 7, 9]))
    assert nbatch.batch_size() == 5
    assert len(nbatch) == 5

def test_edge_batch():
    d = 10
    g = dgl.DGLGraph(nx.path_graph(20))
    nfeat = F.randn((g.number_of_nodes(), d))
    efeat = F.randn((g.number_of_edges(), d))
    g.ndata['x'] = nfeat
    g.edata['x'] = efeat

    # test all
    eid = utils.toindex(slice(0, g.number_of_edges()))
    u, v, _ = g._graph.edges('eid')

    src_data = g.get_n_repr(u)
    edge_data = g.get_e_repr(eid)
    dst_data = g.get_n_repr(v)
    ebatch = EdgeBatch((u, v, eid), src_data, edge_data, dst_data)
    assert F.shape(ebatch.src['x'])[0] == g.number_of_edges() and\
        F.shape(ebatch.src['x'])[1] == d
    assert F.shape(ebatch.dst['x'])[0] == g.number_of_edges() and\
        F.shape(ebatch.dst['x'])[1] == d
    assert F.shape(ebatch.data['x'])[0] == g.number_of_edges() and\
        F.shape(ebatch.data['x'])[1] == d
    assert F.allclose(ebatch.edges()[0], u.tousertensor())
    assert F.allclose(ebatch.edges()[1], v.tousertensor())
    assert F.allclose(ebatch.edges()[2], F.arange(0, g.number_of_edges()))
    assert ebatch.batch_size() == g.number_of_edges()
    assert len(ebatch) == g.number_of_edges()

    # test partial
    eid = utils.toindex(F.tensor([0, 3, 5, 7, 11, 13, 15, 27]))
    u, v, _ = g._graph.find_edges(eid)
    src_data = g.get_n_repr(u)
    edge_data = g.get_e_repr(eid)
    dst_data = g.get_n_repr(v)
    ebatch = EdgeBatch((u, v, eid), src_data, edge_data, dst_data)
    assert F.shape(ebatch.src['x'])[0] == 8 and\
        F.shape(ebatch.src['x'])[1] == d
    assert F.shape(ebatch.dst['x'])[0] == 8 and\
        F.shape(ebatch.dst['x'])[1] == d
    assert F.shape(ebatch.data['x'])[0] == 8 and\
        F.shape(ebatch.data['x'])[1] == d
    assert F.allclose(ebatch.edges()[0], u.tousertensor())
    assert F.allclose(ebatch.edges()[1], v.tousertensor())
    assert F.allclose(ebatch.edges()[2], eid.tousertensor())
    assert ebatch.batch_size() == 8
    assert len(ebatch) == 8

if __name__ == '__main__':
    test_node_batch()
    test_edge_batch()
