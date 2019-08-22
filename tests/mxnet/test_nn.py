import mxnet as mx
import networkx as nx
import numpy as np
import dgl
import dgl.nn.mxnet as nn
import backend as F
from mxnet import autograd, gluon

def check_close(a, b):
    assert np.allclose(a.asnumpy(), b.asnumpy(), rtol=1e-4, atol=1e-4)

def _AXWb(A, X, W, b):
    X = mx.nd.dot(X, W.data(X.context))
    Y = mx.nd.dot(A, X.reshape(X.shape[0], -1)).reshape(X.shape)
    return Y + b.data(X.context)

def test_graph_conv():
    g = dgl.DGLGraph(nx.path_graph(3))
    ctx = F.ctx()
    adj = g.adjacency_matrix(ctx=ctx)

    conv = nn.GraphConv(5, 2, norm=False, bias=True)
    conv.initialize(ctx=ctx)
    # test#1: basic
    h0 = F.ones((3, 5))
    h1 = conv(h0, g)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    check_close(h1, _AXWb(adj, h0, conv.weight, conv.bias))
    # test#2: more-dim
    h0 = F.ones((3, 5, 5))
    h1 = conv(h0, g)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    check_close(h1, _AXWb(adj, h0, conv.weight, conv.bias))

    conv = nn.GraphConv(5, 2)
    conv.initialize(ctx=ctx)

    # test#3: basic
    h0 = F.ones((3, 5))
    h1 = conv(h0, g)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    # test#4: basic
    h0 = F.ones((3, 5, 5))
    h1 = conv(h0, g)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0

    conv = nn.GraphConv(5, 2)
    conv.initialize(ctx=ctx)

    with autograd.train_mode():
        # test#3: basic
        h0 = F.ones((3, 5))
        h1 = conv(h0, g)
        assert len(g.ndata) == 0
        assert len(g.edata) == 0
        # test#4: basic
        h0 = F.ones((3, 5, 5))
        h1 = conv(h0, g)
        assert len(g.ndata) == 0
        assert len(g.edata) == 0

    # test not override features
    g.ndata["h"] = 2 * F.ones((3, 1))
    h1 = conv(h0, g)
    assert len(g.ndata) == 1
    assert len(g.edata) == 0
    assert "h" in g.ndata
    check_close(g.ndata['h'], 2 * F.ones((3, 1)))

def test_set2set():
    g = dgl.DGLGraph(nx.path_graph(10))
    ctx = F.ctx()

    s2s = nn.Set2Set(5, 3, 3) # hidden size 5, 3 iters, 3 layers
    s2s.initialize(ctx=ctx)
    print(s2s)

    # test#1: basic
    h0 = F.randn((g.number_of_nodes(), 5))
    h1 = s2s(h0, g)
    assert h1.shape[0] == 10 and h1.ndim == 1

    # test#2: batched graph
    bg = dgl.batch([g, g, g])
    h0 = F.randn((bg.number_of_nodes(), 5))
    h1 = s2s(h0, bg)
    assert h1.shape[0] == 3 and h1.shape[1] == 10 and h1.ndim == 2

def test_glob_att_pool():
    g = dgl.DGLGraph(nx.path_graph(10))
    ctx = F.ctx()

    gap = nn.GlobalAttentionPooling(gluon.nn.Dense(1), gluon.nn.Dense(10))
    gap.initialize(ctx=ctx)
    print(gap)
    # test#1: basic
    h0 = F.randn((g.number_of_nodes(), 5))
    h1 = gap(h0, g)
    assert h1.shape[0] == 10 and h1.ndim == 1

    # test#2: batched graph
    bg = dgl.batch([g, g, g, g])
    h0 = F.randn((bg.number_of_nodes(), 5))
    h1 = gap(h0, bg)
    assert h1.shape[0] == 4 and h1.shape[1] == 10 and h1.ndim == 2

def test_simple_pool():
    g = dgl.DGLGraph(nx.path_graph(15))

    sum_pool = nn.SumPooling()
    avg_pool = nn.AvgPooling()
    max_pool = nn.MaxPooling()
    sort_pool = nn.SortPooling(10) # k = 10
    print(sum_pool, avg_pool, max_pool, sort_pool)

    # test#1: basic
    h0 = F.randn((g.number_of_nodes(), 5))
    h1 = sum_pool(h0, g)
    check_close(h1, F.sum(h0, 0))
    h1 = avg_pool(h0, g)
    check_close(h1, F.mean(h0, 0))
    h1 = max_pool(h0, g)
    check_close(h1, F.max(h0, 0))
    h1 = sort_pool(h0, g)
    assert h1.shape[0] == 10 * 5 and h1.ndim == 1

    # test#2: batched graph
    g_ = dgl.DGLGraph(nx.path_graph(5))
    bg = dgl.batch([g, g_, g, g_, g])
    h0 = F.randn((bg.number_of_nodes(), 5))
    h1 = sum_pool(h0, bg)
    truth = mx.nd.stack(F.sum(h0[:15], 0),
                        F.sum(h0[15:20], 0),
                        F.sum(h0[20:35], 0),
                        F.sum(h0[35:40], 0),
                        F.sum(h0[40:55], 0), axis=0)
    check_close(h1, truth)

    h1 = avg_pool(h0, bg)
    truth = mx.nd.stack(F.mean(h0[:15], 0),
                        F.mean(h0[15:20], 0),
                        F.mean(h0[20:35], 0),
                        F.mean(h0[35:40], 0),
                        F.mean(h0[40:55], 0), axis=0)
    check_close(h1, truth)

    h1 = max_pool(h0, bg)
    truth = mx.nd.stack(F.max(h0[:15], 0),
                        F.max(h0[15:20], 0),
                        F.max(h0[20:35], 0),
                        F.max(h0[35:40], 0),
                        F.max(h0[40:55], 0), axis=0)
    check_close(h1, truth)

    h1 = sort_pool(h0, bg)
    assert h1.shape[0] == 5 and h1.shape[1] == 10 * 5 and h1.ndim == 2

def uniform_attention(g, shape):
    a = mx.nd.ones(shape)
    target_shape = (g.number_of_edges(),) + (1,) * (len(shape) - 1)
    return a / g.in_degrees(g.edges()[1]).reshape(target_shape).astype('float32')

def test_edge_softmax():
    # Basic
    g = dgl.DGLGraph(nx.path_graph(3))
    edata = F.ones((g.number_of_edges(), 1))
    a = nn.edge_softmax(g, edata)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    assert np.allclose(a.asnumpy(), uniform_attention(g, a.shape).asnumpy(),
            1e-4, 1e-4)

    # Test higher dimension case
    edata = F.ones((g.number_of_edges(), 3, 1))
    a = nn.edge_softmax(g, edata)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    assert np.allclose(a.asnumpy(), uniform_attention(g, a.shape).asnumpy(),
            1e-4, 1e-4)

if __name__ == '__main__':
    test_graph_conv()
    test_edge_softmax()
    test_set2set()
    test_glob_att_pool()
    test_simple_pool()
