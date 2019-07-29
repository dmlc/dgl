import dgl
import backend as F
import networkx as nx

def test_topk_nodes():
    # test#1: basic
    g0 = dgl.DGLGraph(nx.path_graph(14))

    feat0 = F.randn((g0.number_of_nodes(), 10))
    g0.ndata['x'] = feat0
    # to test the case where k > number of nodes.
    dgl.topk_nodes(g0, 'x', 20, idx=-1)
    # test correctness
    val, indices = dgl.topk_nodes(g0, 'x', 5, idx=-1)
    ground_truth = F.reshape(
        F.argsort(F.slice_axis(feat0, -1, 9, 10), 0, True)[:5], (5,))
    assert F.allclose(ground_truth, indices)
    g0.ndata.pop('x')

    # test#2: batched graph
    g1 = dgl.DGLGraph(nx.path_graph(12))
    feat1 = F.randn((g1.number_of_nodes(), 10))

    bg = dgl.batch([g0, g1])
    bg.ndata['x'] = F.cat([feat0, feat1], 0)
    # to test the case where k > number of nodes.
    dgl.topk_nodes(bg, 'x', 16, idx=1)
    # test correctness
    val, indices = dgl.topk_nodes(bg, 'x', 6, descending=False, idx=0)
    ground_truth_0 = F.reshape(
        F.argsort(F.slice_axis(feat0, -1, 0, 1), 0, False)[:6], (6,))
    ground_truth_1 = F.reshape(
        F.argsort(F.slice_axis(feat1, -1, 0, 1), 0, False)[:6], (6,))
    ground_truth = F.stack([ground_truth_0, ground_truth_1], 0)
    assert F.allclose(ground_truth, indices)

def test_topk_edges():
    # test#1: basic
    g0 = dgl.DGLGraph(nx.path_graph(14))

    feat0 = F.randn((g0.number_of_edges(), 10))
    g0.edata['x'] = feat0
    # to test the case where k > number of edges.
    dgl.topk_edges(g0, 'x', 30, idx=-1)
    # test correctness
    val, indices = dgl.topk_edges(g0, 'x', 7, idx=-1)
    ground_truth = F.reshape(
        F.argsort(F.slice_axis(feat0, -1, 9, 10), 0, True)[:7], (7,))
    assert F.allclose(ground_truth, indices)
    g0.edata.pop('x')

    # test#2: batched graph
    g1 = dgl.DGLGraph(nx.path_graph(12))
    feat1 = F.randn((g1.number_of_edges(), 10))

    bg = dgl.batch([g0, g1])
    bg.edata['x'] = F.cat([feat0, feat1], 0)
    # to test the case where k > number of edges.
    dgl.topk_edges(bg, 'x', 33, idx=1)
    # test correctness
    val, indices = dgl.topk_edges(bg, 'x', 4, descending=False, idx=0)
    ground_truth_0 = F.reshape(
        F.argsort(F.slice_axis(feat0, -1, 0, 1), 0, False)[:4], (4,))
    ground_truth_1 = F.reshape(
        F.argsort(F.slice_axis(feat1, -1, 0, 1), 0, False)[:4], (4,))
    ground_truth = F.stack([ground_truth_0, ground_truth_1], 0)
    assert F.allclose(ground_truth, indices)

if __name__ == '__main__':
    test_topk_nodes()
    test_topk_edges()
