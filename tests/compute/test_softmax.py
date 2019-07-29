import dgl
import backend as F
import networkx as nx

def test_softmax_nodes():
    # test#1: basic
    g0 = dgl.DGLGraph(nx.path_graph(9))

    feat0 = F.randn((g0.number_of_nodes(), 10))
    g0.ndata['x'] = feat0
    ground_truth = F.softmax(feat0, dim=0)
    assert F.allclose(dgl.softmax_nodes(g0, 'x'), ground_truth)
    g0.ndata.pop('x')

    # test#2: batched graph
    g1 = dgl.DGLGraph(nx.path_graph(5))
    g2 = dgl.DGLGraph(nx.path_graph(3))
    g3 = dgl.DGLGraph()
    g4 = dgl.DGLGraph(nx.path_graph(10))
    bg = dgl.batch([g0, g1, g2, g3, g4])
    feat1 = F.randn((g1.number_of_nodes(), 10))
    feat2 = F.randn((g2.number_of_nodes(), 10))
    feat4 = F.randn((g4.number_of_nodes(), 10))
    bg.ndata['x'] = F.cat([feat0, feat1, feat2, feat4], 0)
    ground_truth = F.cat([
        F.softmax(feat0, 0),
        F.softmax(feat1, 0),
        F.softmax(feat2, 0),
        F.softmax(feat4, 0)
    ], 0)
    assert F.allclose(dgl.softmax_nodes(bg, 'x'), ground_truth)

def test_softmax_edges():
    # test#1: basic
    g0 = dgl.DGLGraph(nx.path_graph(10))

    feat0 = F.randn((g0.number_of_edges(), 10))
    g0.edata['x'] = feat0
    ground_truth = F.softmax(feat0, dim=0)
    assert F.allclose(dgl.softmax_edges(g0, 'x'), ground_truth)
    g0.edata.pop('x')

    # test#2: batched graph
    g1 = dgl.DGLGraph(nx.path_graph(5))
    g2 = dgl.DGLGraph(nx.path_graph(3))
    g3 = dgl.DGLGraph()
    g4 = dgl.DGLGraph(nx.path_graph(10))
    bg = dgl.batch([g0, g1, g2, g3, g4])
    feat1 = F.randn((g1.number_of_edges(), 10))
    feat2 = F.randn((g2.number_of_edges(), 10))
    feat4 = F.randn((g4.number_of_edges(), 10))
    bg.edata['x'] = F.cat([feat0, feat1, feat2, feat4], 0)
    ground_truth = F.cat([
        F.softmax(feat0, 0),
        F.softmax(feat1, 0),
        F.softmax(feat2, 0),
        F.softmax(feat4, 0)
    ], 0)
    assert F.allclose(dgl.softmax_edges(bg, 'x'), ground_truth)

if __name__ == '__main__':
    test_softmax_nodes()
    test_softmax_edges()