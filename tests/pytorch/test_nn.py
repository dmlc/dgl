import torch as th
import networkx as nx
import dgl
import dgl.nn.pytorch as nn

def _AXWb(A, X, W, b):
    X = th.matmul(X, W)
    Y = th.matmul(A, X.view(X.shape[0], -1)).view_as(X)
    return Y + b

def test_graph_conv():
    g = dgl.DGLGraph(nx.path_graph(3))
    adj = g.adjacency_matrix()

    conv = nn.GraphConv(5, 2, norm=False)
    # test#1: basic
    h0 = th.ones((3, 5))
    h1 = conv(h0, g)
    assert th.allclose(h1, _AXWb(adj, h0, conv.weight, conv.bias))
    # test#2: more-dim
    h0 = th.ones((3, 5, 5))
    h1 = conv(h0, g)
    assert th.allclose(h1, _AXWb(adj, h0, conv.weight, conv.bias))

    conv = nn.GraphConv(5, 2, bias=False)
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

if __name__ == '__main__':
    test_graph_conv()
