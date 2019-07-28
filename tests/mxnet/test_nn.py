import mxnet as mx
import networkx as nx
import numpy as np
import dgl
import dgl.nn.mxnet as nn
from mxnet import autograd, gluon

def check_eq(a, b):
    assert a.shape == b.shape
    assert mx.nd.sum(a == b).asnumpy() == int(np.prod(list(a.shape)))

def _AXWb(A, X, W, b):
    X = mx.nd.dot(X, W.data(X.context))
    Y = mx.nd.dot(A, X.reshape(X.shape[0], -1)).reshape(X.shape)
    return Y + b.data(X.context)

def test_graph_conv():
    g = dgl.DGLGraph(nx.path_graph(3))
    adj = g.adjacency_matrix()
    ctx = mx.cpu(0)

    conv = nn.GraphConv(5, 2, norm=False, bias=True)
    conv.initialize(ctx=ctx)
    # test#1: basic
    h0 = mx.nd.ones((3, 5))
    h1 = conv(h0, g)
    check_eq(h1, _AXWb(adj, h0, conv.weight, conv.bias))
    # test#2: more-dim
    h0 = mx.nd.ones((3, 5, 5))
    h1 = conv(h0, g)
    check_eq(h1, _AXWb(adj, h0, conv.weight, conv.bias))

    conv = nn.GraphConv(5, 2)
    conv.initialize(ctx=ctx)

    # test#3: basic
    h0 = mx.nd.ones((3, 5))
    h1 = conv(h0, g)
    # test#4: basic
    h0 = mx.nd.ones((3, 5, 5))
    h1 = conv(h0, g)

    conv = nn.GraphConv(5, 2)
    conv.initialize(ctx=ctx)

    with autograd.train_mode():
        # test#3: basic
        h0 = mx.nd.ones((3, 5))
        h1 = conv(h0, g)
        # test#4: basic
        h0 = mx.nd.ones((3, 5, 5))
        h1 = conv(h0, g)

    # test repeated features
    g.ndata["_gconv_feat"] = 2 * mx.nd.ones((3, 1))
    h1 = conv(h0, g)
    assert "_gconv_feat" in g.ndata

def test_set2set():
    g = dgl.DGLGraph(nx.path_graph(10))

    s2s = nn.Set2Set(5, 3, 3) # hidden size 5, 3 iters, 3 layers
    print(s2s)

    # test#1: basic
    h0 = mx.nd.random.randn(g.number_of_nodes(), 5)
    h1 = s2s(h0, g)
    print(h1)

    # test#2: batched graph
    bg = dgl.batch([g, g, g])
    h0 = mx.nd.random.randn(bg.number_of_nodes(), 5)
    h1 = s2s(h0, bg)
    print(h1)

def test_glob_att_pool():
    g = dgl.DGLGraph(nx.path_graph(10))

    gap = nn.GlobAttnPooling(gluon.nn.Dense(1), gluon.nn.Dense(10))
    print(gap)
    # test#1: basic
    h0 = mx.nd.random.randn(g.number_of_nodes(), 5)
    h1 = gap(h0, g)
    print(h1)

    # test#2: batched graph
    bg = dgl.batch([g, g, g, g])
    h0 = mx.nd.random.randn(bg.number_of_nodes(), 5)
    h1 = gap(h0, bg)
    print(h1)


def test_simple_pool():
    g = dgl.DGLGraph(nx.path_graph(15))

    sum_pool = nn.SumPooling()
    avg_pool = nn.AvgPooling()
    max_pool = nn.MaxPooling()
    sort_pool = nn.SortPooling(10) # k = 10
    print(sum_pool, avg_pool, max_pool, sort_pool)

    # test#1: basic
    h0 = mx.nd.random.randn(g.number_of_nodes(), 5)
    h1 = sum_pool(h0, g)
    print(h1)
    h1 = avg_pool(h0, g)
    print(h1)
    h1 = max_pool(h0, g)
    print(h1)
    h1 = sort_pool(h0, g)
    print(h1)

    # test#2: batched graph
    g_ = dgl.DGLGraph(nx.path_graph(5))
    bg = dgl.batch([g, g_, g, g_, g])
    h0 = mx.nd.random.randn(bg.number_of_nodes(), 5)
    h1 = sum_pool(h0, bg)
    print(h1)
    h1 = avg_pool(h0, bg)
    print(h1)
    h1 = max_pool(h0, bg)
    print(h1)
    h1 = sort_pool(h0, bg)
    print(h1)


def uniform_attention(g, shape):
    a = mx.nd.ones(shape)
    target_shape = (g.number_of_edges(),) + (1,) * (len(shape) - 1)
    return a / g.in_degrees(g.edges()[1]).reshape(target_shape).astype('float32')

def test_edge_softmax():
    # Basic
    g = dgl.DGLGraph(nx.path_graph(3))
    edata = mx.nd.ones((g.number_of_edges(), 1))
    a = nn.edge_softmax(g, edata)
    assert np.allclose(a.asnumpy(), uniform_attention(g, a.shape).asnumpy(),
            1e-4, 1e-4)

    # Test higher dimension case
    edata = mx.nd.ones((g.number_of_edges(), 3, 1))
    a = nn.edge_softmax(g, edata)
    assert np.allclose(a.asnumpy(), uniform_attention(g, a.shape).asnumpy(),
            1e-4, 1e-4)

if __name__ == '__main__':
    test_graph_conv()
    test_edge_softmax()
    test_set2set()
    test_glob_att_pool()
    test_simple_pool()
