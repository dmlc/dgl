import dgl
import backend as F
import networkx as nx

def test_broadcast_node():
    # test#1: basic
    g0 = dgl.DGLGraph(nx.path_graph(10))
    feat0 = F.randn((40,))
    ground_truth = F.stack([feat0] * g0.number_of_nodes(), 0)
    assert F.allclose(dgl.broadcast_nodes(g0, feat0), ground_truth)

    # test#2: batched graph
    g1 = dgl.DGLGraph(nx.path_graph(3))
    g2 = dgl.DGLGraph()
    g3 = dgl.DGLGraph(nx.path_graph(12))
    bg = dgl.batch([g0, g1, g2, g3])
    feat1 = F.randn((40,))
    feat2 = F.randn((40,))
    feat3 = F.randn((40,))
    ground_truth = F.stack(
        [feat0] * g0.number_of_nodes() +\
        [feat1] * g1.number_of_nodes() +\
        [feat2] * g2.number_of_nodes() +\
        [feat3] * g3.number_of_nodes(), 0
    )
    assert F.allclose(dgl.broadcast_nodes(
        bg, F.stack([feat0, feat1, feat2, feat3], 0)
    ), ground_truth)

def test_broadcast_edge():
    # test#1: basic
    g0 = dgl.DGLGraph(nx.path_graph(10))
    feat0 = F.randn((40,))
    ground_truth = F.stack([feat0] * g0.number_of_edges(), 0)
    assert F.allclose(dgl.broadcast_edges(g0, feat0), ground_truth)

    # test#2: batched graph
    g1 = dgl.DGLGraph(nx.path_graph(3))
    g2 = dgl.DGLGraph()
    g3 = dgl.DGLGraph(nx.path_graph(12))
    bg = dgl.batch([g0, g1, g2, g3])
    feat1 = F.randn((40,))
    feat2 = F.randn((40,))
    feat3 = F.randn((40,))
    ground_truth = F.stack(
        [feat0] * g0.number_of_edges() +\
        [feat1] * g1.number_of_edges() +\
        [feat2] * g2.number_of_edges() +\
        [feat3] * g3.number_of_edges(), 0
    )
    assert F.allclose(dgl.broadcast_edges(
        bg, F.stack([feat0, feat1, feat2, feat3], 0)
    ), ground_truth)

if __name__ == '__main__':
    test_broadcast_node()
    test_broadcast_edge()