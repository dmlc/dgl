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

if __name__ == '__main__':
    test_graph_conv()
