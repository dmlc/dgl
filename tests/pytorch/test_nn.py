import torch as th
import networkx as nx
import dgl
import dgl.nn.pytorch as nn
from copy import deepcopy

import numpy as np
import scipy as sp

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
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    assert th.allclose(h1, _AXWb(adj, h0, conv.weight, conv.bias))
    # test#2: more-dim
    h0 = th.ones((3, 5, 5))
    h1 = conv(h0, g)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    assert th.allclose(h1, _AXWb(adj, h0, conv.weight, conv.bias))

    conv = nn.GraphConv(5, 2)
    # test#3: basic
    h0 = th.ones((3, 5))
    h1 = conv(h0, g)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    # test#4: basic
    h0 = th.ones((3, 5, 5))
    h1 = conv(h0, g)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0

    conv = nn.GraphConv(5, 2)
    # test#3: basic
    h0 = th.ones((3, 5))
    h1 = conv(h0, g)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    # test#4: basic
    h0 = th.ones((3, 5, 5))
    h1 = conv(h0, g)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0

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
    # Basic
    g = dgl.DGLGraph(nx.path_graph(3))
    edata = th.ones(g.number_of_edges(), 1)
    a = nn.edge_softmax(g, edata)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    assert th.allclose(a, uniform_attention(g, a.shape))

    # Test higher dimension case
    edata = th.ones(g.number_of_edges(), 3, 1)
    a = nn.edge_softmax(g, edata)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    assert th.allclose(a, uniform_attention(g, a.shape))

    # Test both forward and backward with PyTorch built-in softmax.
    g = dgl.DGLGraph()
    g.add_nodes(30)
    # build a complete graph
    for i in range(30):
        for j in range(30):
            g.add_edge(i, j)

    score = th.rand(900, 1)
    score.requires_grad_()
    grad = th.rand(900, 1)
    y = th.softmax(score.view(30, 30), dim=0).view(-1, 1)
    y.backward(grad)
    grad_score = score.grad
    score.grad.zero_()
    y_dgl = nn.edge_softmax(g, score)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    # check forward
    assert th.allclose(y_dgl, y)
    y_dgl.backward(grad)
    # checkout gradient
    assert th.allclose(score.grad, grad_score)
    print(score.grad[:10], grad_score[:10])
    
    # Test 2
    def generate_rand_graph(n):
      arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
      return dgl.DGLGraph(arr, readonly=True)
    
    g = generate_rand_graph(50)
    a1 = th.randn(g.number_of_edges(), 1).requires_grad_()
    a2 = a1.clone().detach().requires_grad_()
    g.edata['s'] = a1
    g.group_apply_edges('dst', lambda edges: {'ss':th.softmax(edges.data['s'], 1)})
    g.edata['ss'].sum().backward()
    
    builtin_sm = nn.edge_softmax(g, a2)
    builtin_sm.sum().backward()
    print(a1.grad - a2.grad)
    assert len(g.ndata) == 0
    assert len(g.edata) == 2
    assert th.allclose(a1.grad, a2.grad, rtol=1e-4, atol=1e-4) # Follow tolerance in unittest backend
    

if __name__ == '__main__':
    test_graph_conv()
    test_edge_softmax()
