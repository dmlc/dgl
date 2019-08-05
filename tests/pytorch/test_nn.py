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

def test_set2set():
    g = dgl.DGLGraph(nx.path_graph(10))

    s2s = nn.Set2Set(5, 3, 3) # hidden size 5, 3 iters, 3 layers
    print(s2s)

    # test#1: basic
    h0 = th.rand(g.number_of_nodes(), 5)
    h1 = s2s(h0, g)
    assert h1.shape[0] == 10 and h1.dim() == 1

    # test#2: batched graph
    g1 = dgl.DGLGraph(nx.path_graph(11))
    g2 = dgl.DGLGraph(nx.path_graph(5))
    bg = dgl.batch([g, g1, g2])
    h0 = th.rand(bg.number_of_nodes(), 5)
    h1 = s2s(h0, bg)
    assert h1.shape[0] == 3 and h1.shape[1] == 10 and h1.dim() == 2

def test_glob_att_pool():
    g = dgl.DGLGraph(nx.path_graph(10))

    gap = nn.GlobalAttentionPooling(th.nn.Linear(5, 1), th.nn.Linear(5, 10))
    print(gap)

    # test#1: basic
    h0 = th.rand(g.number_of_nodes(), 5)
    h1 = gap(h0, g)
    assert h1.shape[0] == 10 and h1.dim() == 1

    # test#2: batched graph
    bg = dgl.batch([g, g, g, g])
    h0 = th.rand(bg.number_of_nodes(), 5)
    h1 = gap(h0, bg)
    assert h1.shape[0] == 4 and h1.shape[1] == 10 and h1.dim() == 2

def test_simple_pool():
    g = dgl.DGLGraph(nx.path_graph(15))

    sum_pool = nn.SumPooling()
    avg_pool = nn.AvgPooling()
    max_pool = nn.MaxPooling()
    sort_pool = nn.SortPooling(10) # k = 10
    print(sum_pool, avg_pool, max_pool, sort_pool)

    # test#1: basic
    h0 = th.rand(g.number_of_nodes(), 5)
    h1 = sum_pool(h0, g)
    assert th.allclose(h1, th.sum(h0, 0))
    h1 = avg_pool(h0, g)
    assert th.allclose(h1, th.mean(h0, 0))
    h1 = max_pool(h0, g)
    assert th.allclose(h1, th.max(h0, 0)[0])
    h1 = sort_pool(h0, g)
    assert h1.shape[0] == 10 * 5 and h1.dim() == 1

    # test#2: batched graph
    g_ = dgl.DGLGraph(nx.path_graph(5))
    bg = dgl.batch([g, g_, g, g_, g])
    h0 = th.rand(bg.number_of_nodes(), 5)

    h1 = sum_pool(h0, bg)
    truth = th.stack([th.sum(h0[:15], 0),
                      th.sum(h0[15:20], 0),
                      th.sum(h0[20:35], 0),
                      th.sum(h0[35:40], 0),
                      th.sum(h0[40:55], 0)], 0)
    assert th.allclose(h1, truth)

    h1 = avg_pool(h0, bg)
    truth = th.stack([th.mean(h0[:15], 0),
                      th.mean(h0[15:20], 0),
                      th.mean(h0[20:35], 0),
                      th.mean(h0[35:40], 0),
                      th.mean(h0[40:55], 0)], 0)
    assert th.allclose(h1, truth)

    h1 = max_pool(h0, bg)
    truth = th.stack([th.max(h0[:15], 0)[0],
                      th.max(h0[15:20], 0)[0],
                      th.max(h0[20:35], 0)[0],
                      th.max(h0[35:40], 0)[0],
                      th.max(h0[40:55], 0)[0]], 0)
    assert th.allclose(h1, truth)

    h1 = sort_pool(h0, bg)
    assert h1.shape[0] == 5 and h1.shape[1] == 10 * 5 and h1.dim() == 2

def test_set_trans():
    g = dgl.DGLGraph(nx.path_graph(15))

    st_enc_0 = nn.SetTransformerEncoder(50, 5, 10, 100, 2, 'sab')
    st_enc_1 = nn.SetTransformerEncoder(50, 5, 10, 100, 2, 'isab', 3)
    st_dec = nn.SetTransformerDecoder(50, 5, 10, 100, 2, 4)
    print(st_enc_0, st_enc_1, st_dec)

    # test#1: basic
    h0 = th.rand(g.number_of_nodes(), 50)
    h1 = st_enc_0(h0, g)
    assert h1.shape == h0.shape
    h1 = st_enc_1(h0, g)
    assert h1.shape == h0.shape
    h2 = st_dec(h1, g)
    assert h2.shape[0] == 200 and h2.dim() == 1

    # test#2: batched graph
    g1 = dgl.DGLGraph(nx.path_graph(5))
    g2 = dgl.DGLGraph(nx.path_graph(10))
    bg = dgl.batch([g, g1, g2])
    h0 = th.rand(bg.number_of_nodes(), 50)
    h1 = st_enc_0(h0, bg)
    assert h1.shape == h0.shape
    h1 = st_enc_1(h0, bg)
    assert h1.shape == h0.shape

    h2 = st_dec(h1, bg)
    assert h2.shape[0] == 3 and h2.shape[1] == 200 and h2.dim() == 2

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
    test_set2set()
    test_glob_att_pool()
    test_simple_pool()
    test_set_trans()
