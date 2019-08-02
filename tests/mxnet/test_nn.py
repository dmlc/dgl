import mxnet as mx
import networkx as nx
import numpy as np
import dgl
import dgl.nn.mxnet as nn
from mxnet import autograd

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
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    check_eq(h1, _AXWb(adj, h0, conv.weight, conv.bias))
    # test#2: more-dim
    h0 = mx.nd.ones((3, 5, 5))
    h1 = conv(h0, g)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    check_eq(h1, _AXWb(adj, h0, conv.weight, conv.bias))

    conv = nn.GraphConv(5, 2)
    conv.initialize(ctx=ctx)

    # test#3: basic
    h0 = mx.nd.ones((3, 5))
    h1 = conv(h0, g)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    # test#4: basic
    h0 = mx.nd.ones((3, 5, 5))
    h1 = conv(h0, g)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0

    conv = nn.GraphConv(5, 2)
    conv.initialize(ctx=ctx)

    with autograd.train_mode():
        # test#3: basic
        h0 = mx.nd.ones((3, 5))
        h1 = conv(h0, g)
        assert len(g.ndata) == 0
        assert len(g.edata) == 0
        # test#4: basic
        h0 = mx.nd.ones((3, 5, 5))
        h1 = conv(h0, g)
        assert len(g.ndata) == 0
        assert len(g.edata) == 0

    # test not override features
    g.ndata["h"] = 2 * mx.nd.ones((3, 1))
    h1 = conv(h0, g)
    assert len(g.ndata) == 1
    assert len(g.edata) == 0
    assert "h" in g.ndata
    check_eq(g.ndata['h'], 2 * mx.nd.ones((3, 1)))

def uniform_attention(g, shape):
    a = mx.nd.ones(shape)
    target_shape = (g.number_of_edges(),) + (1,) * (len(shape) - 1)
    return a / g.in_degrees(g.edges()[1]).reshape(target_shape).astype('float32')

def test_edge_softmax():
    # Basic
    g = dgl.DGLGraph(nx.path_graph(3))
    edata = mx.nd.ones((g.number_of_edges(), 1))
    a = nn.edge_softmax(g, edata)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    assert np.allclose(a.asnumpy(), uniform_attention(g, a.shape).asnumpy(),
            1e-4, 1e-4)

    # Test higher dimension case
    edata = mx.nd.ones((g.number_of_edges(), 3, 1))
    a = nn.edge_softmax(g, edata)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    assert np.allclose(a.asnumpy(), uniform_attention(g, a.shape).asnumpy(),
            1e-4, 1e-4)

if __name__ == '__main__':
    test_graph_conv()
    test_edge_softmax()
