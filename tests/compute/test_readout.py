import dgl
import backend as F
import networkx as nx
import unittest

def test_simple_readout():
    g1 = dgl.DGLGraph()
    g1.add_nodes(3)
    g2 = dgl.DGLGraph()
    g2.add_nodes(4) # no edges
    g1.add_edges([0, 1, 2], [2, 0, 1])

    n1 = F.randn((3, 5))
    n2 = F.randn((4, 5))
    e1 = F.randn((3, 5))
    s1 = F.sum(n1, 0)   # node sums
    s2 = F.sum(n2, 0)
    se1 = F.sum(e1, 0)  # edge sums
    m1 = F.mean(n1, 0)  # node means
    m2 = F.mean(n2, 0)
    me1 = F.mean(e1, 0) # edge means
    w1 = F.randn((3,))
    w2 = F.randn((4,))
    max1 = F.max(n1, 0)
    max2 = F.max(n2, 0)
    maxe1 = F.max(e1, 0)
    ws1 = F.sum(n1 * F.unsqueeze(w1, 1), 0)
    ws2 = F.sum(n2 * F.unsqueeze(w2, 1), 0)
    wm1 = F.sum(n1 * F.unsqueeze(w1, 1), 0) / F.sum(F.unsqueeze(w1, 1), 0)
    wm2 = F.sum(n2 * F.unsqueeze(w2, 1), 0) / F.sum(F.unsqueeze(w2, 1), 0)
    g1.ndata['x'] = n1
    g2.ndata['x'] = n2
    g1.ndata['w'] = w1
    g2.ndata['w'] = w2
    g1.edata['x'] = e1

    assert F.allclose(dgl.sum_nodes(g1, 'x'), s1)
    assert F.allclose(dgl.sum_nodes(g1, 'x', 'w'), ws1)
    assert F.allclose(dgl.sum_edges(g1, 'x'), se1)
    assert F.allclose(dgl.mean_nodes(g1, 'x'), m1)
    assert F.allclose(dgl.mean_nodes(g1, 'x', 'w'), wm1)
    assert F.allclose(dgl.mean_edges(g1, 'x'), me1)
    assert F.allclose(dgl.max_nodes(g1, 'x'), max1)
    assert F.allclose(dgl.max_edges(g1, 'x'), maxe1)

    g = dgl.batch([g1, g2])
    s = dgl.sum_nodes(g, 'x')
    m = dgl.mean_nodes(g, 'x')
    max_bg = dgl.max_nodes(g, 'x')
    assert F.allclose(s, F.stack([s1, s2], 0))
    assert F.allclose(m, F.stack([m1, m2], 0))
    assert F.allclose(max_bg, F.stack([max1, max2], 0))
    ws = dgl.sum_nodes(g, 'x', 'w')
    wm = dgl.mean_nodes(g, 'x', 'w')
    assert F.allclose(ws, F.stack([ws1, ws2], 0))
    assert F.allclose(wm, F.stack([wm1, wm2], 0))
    s = dgl.sum_edges(g, 'x')
    m = dgl.mean_edges(g, 'x')
    max_bg_e = dgl.max_edges(g, 'x')
    assert F.allclose(s, F.stack([se1, F.zeros(5)], 0))
    assert F.allclose(m, F.stack([me1, F.zeros(5)], 0))
    # TODO(zihao): fix -inf issue
    # assert F.allclose(max_bg_e, F.stack([maxe1, F.zeros(5)], 0)) 


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

    # test idx=None
    val, indices = dgl.topk_nodes(bg, 'x', 6, descending=True)
    assert F.allclose(val, F.stack([F.topk(feat0, 6, 0), F.topk(feat1, 6, 0)], 0))


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

    # test idx=None
    val, indices = dgl.topk_edges(bg, 'x', 6, descending=True)
    assert F.allclose(val, F.stack([F.topk(feat0, 6, 0), F.topk(feat1, 6, 0)], 0))

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

def test_broadcast_nodes():
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

def test_broadcast_edges():
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
    test_simple_readout()
    test_topk_nodes()
    test_topk_edges()
    test_softmax_nodes()
    test_softmax_edges()
    test_broadcast_nodes()
    test_broadcast_edges()
