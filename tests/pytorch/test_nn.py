import torch as th
import networkx as nx
import dgl
import dgl.nn.pytorch as nn
from copy import deepcopy

def _AXWb(A, X, W, b):
    X = th.matmul(X, W)
    Y = th.matmul(A, X.view(X.shape[0], -1)).view_as(X)
    return Y + b

def test_graph_conv():
    g = dgl.DGLGraph(nx.path_graph(3))
    adj = g.adjacency_matrix()

    conv = nn.GraphConv(5, 2, norm=False, bias=True, activation=None)
    # test#1: basic
    h0 = th.ones((3, 5))
    h1 = conv(h0, g)
    assert th.allclose(h1, _AXWb(adj, h0, conv.weight, conv.bias))
    # test#2: more-dim
    h0 = th.ones((3, 5, 5))
    h1 = conv(h0, g)
    assert th.allclose(h1, _AXWb(adj, h0, conv.weight, conv.bias))

    conv = nn.GraphConv(5, 2)
    # test#3: basic
    h0 = th.ones((3, 5))
    h1 = conv(h0, g)
    # test#4: basic
    h0 = th.ones((3, 5, 5))
    h1 = conv(h0, g)

    conv = nn.GraphConv(5, 2, dropout=0.5)
    # test#3: basic
    h0 = th.ones((3, 5))
    h1 = conv(h0, g)
    # test#4: basic
    h0 = th.ones((3, 5, 5))
    h1 = conv(h0, g)

    # test rest_parameters
    old_weight = deepcopy(conv.weight.data)
    conv.reset_parameters()
    new_weight = conv.weight.data
    assert not th.allclose(old_weight, new_weight)

    # test repeated features
    g.ndata["_gconv_feat"] = 2 * th.ones(3, 1)
    h1 = conv(h0, g)
    assert "_gconv_feat" in g.ndata

def uniform_attention(g, shape):
    a = th.ones(shape)
    target_shape = (g.number_of_edges(),) + (1,) * (len(shape) - 1)
    return a / g.in_degrees(g.edges()[1]).view(target_shape).float()

def test_edge_softmax():
    edge_softmax = nn.EdgeSoftmax("a", "max_a", "sum_a")

    # Basic
    g = dgl.DGLGraph(nx.path_graph(3))
    edata = th.ones(g.number_of_edges(), 1)
    unnormalized, normalizer = edge_softmax(edata, g)
    g.edata["a"] = unnormalized
    g.ndata["a_sum"] = normalizer
    g.apply_edges(lambda edges : {"a": edges.data["a"] / edges.dst["a_sum"]})
    assert th.allclose(g.edata["a"], uniform_attention(g, unnormalized.shape))

    # Test repeated features
    g.edata["a"] = 2 * th.ones(4, 1)
    g.ndata["max_a"] = 3 * th.ones(3, 1)
    g.ndata["sum_a"] = 4 * th.ones(3, 1)
    _, _ = edge_softmax(edata, g)
    assert "a" in g.edata
    assert "max_a" in g.ndata
    assert "max_a0" in g.ndata
    assert "sum_a" in g.ndata

    # Test higher dimension case
    edata = th.ones(g.number_of_edges(), 3, 1)
    unnormalized, normalizer = edge_softmax(edata, g)
    g.edata["a"] = unnormalized
    g.ndata["a_sum"] = normalizer
    g.apply_edges(lambda edges : {"a": edges.data["a"] / edges.dst["a_sum"]})
    assert th.allclose(g.edata["a"], uniform_attention(g, unnormalized.shape))

if __name__ == '__main__':
    test_graph_conv()
    test_edge_softmax()
