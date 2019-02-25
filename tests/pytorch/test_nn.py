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

    conv = nn.GraphConv(5, 2, norm=False, bias=True)
    print(conv)
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

    conv = nn.GraphConv(5, 2)
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

def uniform_attention(g, shape):
    a = th.ones(shape)
    target_shape = (g.number_of_edges(),) + (1,) * (len(shape) - 1)
    return a / g.in_degrees(g.edges()[1]).view(target_shape).float()

def test_edge_softmax():
    edge_softmax = nn.EdgeSoftmax()
    print(edge_softmax)

    # Basic
    g = dgl.DGLGraph(nx.path_graph(3))
    edata = th.ones(g.number_of_edges(), 1)
    unnormalized, normalizer = edge_softmax(edata, g)
    g.edata["a"] = unnormalized
    g.ndata["a_sum"] = normalizer
    g.apply_edges(lambda edges : {"a": edges.data["a"] / edges.dst["a_sum"]})
    assert th.allclose(g.edata["a"], uniform_attention(g, unnormalized.shape))

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
