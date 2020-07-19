import dgl
import backend as F
import networkx as nx
import unittest
import pytest
from test_utils.graph_cases import get_cases
from utils import parametrize_dtype

@parametrize_dtype
def test_sum_case1(idtype):
    # NOTE: If you want to update this test case, remember to update the docstring
    #  example too!!!
    g1 = dgl.graph(([0, 1], [1, 0]), idtype=idtype, device=F.ctx())
    g1.ndata['h'] = F.tensor([1., 2.])
    g2 = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    g2.ndata['h'] = F.tensor([1., 2., 3.])
    bg = dgl.batch([g1, g2])
    bg.ndata['w'] = F.tensor([.1, .2, .1, .5, .2])
    assert F.allclose(F.tensor([3.]), dgl.sum_nodes(g1, 'h'))
    assert F.allclose(F.tensor([3., 6.]), dgl.sum_nodes(bg, 'h'))
    assert F.allclose(F.tensor([.5, 1.7]), dgl.sum_nodes(bg, 'h', 'w'))

@parametrize_dtype
@pytest.mark.parametrize('g', get_cases(['homo'], exclude=['dglgraph']))
@pytest.mark.parametrize('reducer', ['sum', 'max', 'mean'])
def test_reduce_readout(g, idtype, reducer):
    g = g.astype(idtype).to(F.ctx())
    g.ndata['h'] = F.randn((g.number_of_nodes(), 3))
    g.edata['h'] = F.randn((g.number_of_edges(), 2))

    # Test.1: node readout
    x = getattr(dgl, '{}_nodes'.format(reducer))(g, 'h')
    # check correctness
    subg = dgl.unbatch(g)
    subx = []
    for sg in subg:
        sx = getattr(dgl, '{}_nodes'.format(reducer))(sg, 'h')
        subx.append(sx)
    assert F.allclose(x, F.cat(subx, dim=0))

    # Test.2: edge readout
    x = getattr(dgl, '{}_edges'.format(reducer))(g, 'h')
    # check correctness
    subg = dgl.unbatch(g)
    subx = []
    for sg in subg:
        sx = getattr(dgl, '{}_edges'.format(reducer))(sg, 'h')
        subx.append(sx)
    assert F.allclose(x, F.cat(subx, dim=0))

@parametrize_dtype
@pytest.mark.parametrize('g', get_cases(['homo'], exclude=['dglgraph']))
@pytest.mark.parametrize('reducer', ['sum', 'max', 'mean'])
def test_weighted_reduce_readout(g, idtype, reducer):
    g = g.astype(idtype).to(F.ctx())
    g.ndata['h'] = F.randn((g.number_of_nodes(), 3))
    g.ndata['w'] = F.randn((g.number_of_nodes(), 1))
    g.edata['h'] = F.randn((g.number_of_edges(), 2))
    g.edata['w'] = F.randn((g.number_of_edges(), 1))

    # Test.1: node readout
    x = getattr(dgl, '{}_nodes'.format(reducer))(g, 'h', 'w')
    # check correctness
    subg = dgl.unbatch(g)
    subx = []
    for sg in subg:
        sx = getattr(dgl, '{}_nodes'.format(reducer))(sg, 'h', 'w')
        subx.append(sx)
    assert F.allclose(x, F.cat(subx, dim=0))

    # Test.2: edge readout
    x = getattr(dgl, '{}_edges'.format(reducer))(g, 'h', 'w')
    # check correctness
    subg = dgl.unbatch(g)
    subx = []
    for sg in subg:
        sx = getattr(dgl, '{}_edges'.format(reducer))(sg, 'h', 'w')
        subx.append(sx)
    assert F.allclose(x, F.cat(subx, dim=0))

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
    feat0 = F.randn((1, 40))
    ground_truth = F.stack([feat0] * g0.number_of_nodes(), 0)
    assert F.allclose(dgl.broadcast_nodes(g0, feat0), ground_truth)

    # test#2: batched graph
    g1 = dgl.DGLGraph(nx.path_graph(3))
    g2 = dgl.DGLGraph()
    g3 = dgl.DGLGraph(nx.path_graph(12))
    bg = dgl.batch([g0, g1, g2, g3])
    feat1 = F.randn((1, 40))
    feat2 = F.randn((1, 40))
    feat3 = F.randn((1, 40))
    ground_truth = F.cat(
        [feat0] * g0.number_of_nodes() +\
        [feat1] * g1.number_of_nodes() +\
        [feat2] * g2.number_of_nodes() +\
        [feat3] * g3.number_of_nodes(), 0
    )
    assert F.allclose(dgl.broadcast_nodes(
        bg, F.cat([feat0, feat1, feat2, feat3], 0)
    ), ground_truth)

def test_broadcast_edges():
    # test#1: basic
    g0 = dgl.DGLGraph(nx.path_graph(10))
    feat0 = F.randn((1, 40))
    ground_truth = F.stack([feat0] * g0.number_of_edges(), 0)
    assert F.allclose(dgl.broadcast_edges(g0, feat0), ground_truth)

    # test#2: batched graph
    g1 = dgl.DGLGraph(nx.path_graph(3))
    g2 = dgl.DGLGraph()
    g3 = dgl.DGLGraph(nx.path_graph(12))
    bg = dgl.batch([g0, g1, g2, g3])
    feat1 = F.randn((1, 40))
    feat2 = F.randn((1, 40))
    feat3 = F.randn((1, 40))
    ground_truth = F.cat(
        [feat0] * g0.number_of_edges() +\
        [feat1] * g1.number_of_edges() +\
        [feat2] * g2.number_of_edges() +\
        [feat3] * g3.number_of_edges(), 0
    )
    assert F.allclose(dgl.broadcast_edges(
        bg, F.cat([feat0, feat1, feat2, feat3], 0)
    ), ground_truth)

if __name__ == '__main__':
    test_simple_readout()
    test_topk_nodes()
    test_topk_edges()
    test_softmax_nodes()
    test_softmax_edges()
    test_broadcast_nodes()
    test_broadcast_edges()
