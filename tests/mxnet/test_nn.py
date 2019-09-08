import mxnet as mx
import networkx as nx
import numpy as np
import scipy as sp
import dgl
import dgl.nn.mxnet as nn
import backend as F
from mxnet import autograd, gluon, nd

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
    h1 = conv(g, h0)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    check_close(h1, _AXWb(adj, h0, conv.weight, conv.bias))
    # test#2: more-dim
    h0 = F.ones((3, 5, 5))
    h1 = conv(g, h0)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    check_close(h1, _AXWb(adj, h0, conv.weight, conv.bias))

    conv = nn.GraphConv(5, 2)
    conv.initialize(ctx=ctx)

    # test#3: basic
    h0 = F.ones((3, 5))
    h1 = conv(g, h0)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    # test#4: basic
    h0 = F.ones((3, 5, 5))
    h1 = conv(g, h0)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0

    conv = nn.GraphConv(5, 2)
    conv.initialize(ctx=ctx)

    with autograd.train_mode():
        # test#3: basic
        h0 = F.ones((3, 5))
        h1 = conv(g, h0)
        assert len(g.ndata) == 0
        assert len(g.edata) == 0
        # test#4: basic
        h0 = F.ones((3, 5, 5))
        h1 = conv(g, h0)
        assert len(g.ndata) == 0
        assert len(g.edata) == 0

    # test not override features
    g.ndata["h"] = 2 * F.ones((3, 1))
    h1 = conv(g, h0)
    assert len(g.ndata) == 1
    assert len(g.edata) == 0
    assert "h" in g.ndata
    check_close(g.ndata['h'], 2 * F.ones((3, 1)))

def _S2AXWb(A, N, X, W, b):
    X1 = X * N
    X1 = mx.nd.dot(A, X1.reshape(X1.shape[0], -1))
    X1 = X1 * N
    X2 = X1 * N
    X2 = mx.nd.dot(A, X2.reshape(X2.shape[0], -1))
    X2 = X2 * N
    X = mx.nd.concat(X, X1, X2, dim=-1)
    Y = mx.nd.dot(X, W)

    return Y + b

def test_tagconv():
    g = dgl.DGLGraph(nx.path_graph(3))
    ctx = F.ctx()
    adj = g.adjacency_matrix(ctx=ctx)
    norm = mx.nd.power(g.in_degrees().astype('float32'), -0.5)

    conv = nn.TAGConv(5, 2, bias=True)
    conv.initialize(ctx=ctx)
    print(conv)

    # test#1: basic
    h0 = F.ones((3, 5))
    h1 = conv(g, h0)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    shp = norm.shape + (1,) * (h0.ndim - 1)
    norm = norm.reshape(shp).as_in_context(h0.context)

    assert F.allclose(h1, _S2AXWb(adj, norm, h0, conv.lin.data(ctx), conv.h_bias.data(ctx)))

    conv = nn.TAGConv(5, 2)
    conv.initialize(ctx=ctx)

    # test#2: basic
    h0 = F.ones((3, 5))
    h1 = conv(g, h0)
    assert h1.shape[-1] == 2

def test_set2set():
    g = dgl.DGLGraph(nx.path_graph(10))
    ctx = F.ctx()

    s2s = nn.Set2Set(5, 3, 3) # hidden size 5, 3 iters, 3 layers
    s2s.initialize(ctx=ctx)
    print(s2s)

    # test#1: basic
    h0 = F.randn((g.number_of_nodes(), 5))
    h1 = s2s(g, h0)
    assert h1.shape[0] == 10 and h1.ndim == 1

    # test#2: batched graph
    bg = dgl.batch([g, g, g])
    h0 = F.randn((bg.number_of_nodes(), 5))
    h1 = s2s(bg, h0)
    assert h1.shape[0] == 3 and h1.shape[1] == 10 and h1.ndim == 2

def test_glob_att_pool():
    g = dgl.DGLGraph(nx.path_graph(10))
    ctx = F.ctx()

    gap = nn.GlobalAttentionPooling(gluon.nn.Dense(1), gluon.nn.Dense(10))
    gap.initialize(ctx=ctx)
    print(gap)
    # test#1: basic
    h0 = F.randn((g.number_of_nodes(), 5))
    h1 = gap(g, h0)
    assert h1.shape[0] == 10 and h1.ndim == 1

    # test#2: batched graph
    bg = dgl.batch([g, g, g, g])
    h0 = F.randn((bg.number_of_nodes(), 5))
    h1 = gap(bg, h0)
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
    h1 = sum_pool(g, h0)
    check_close(h1, F.sum(h0, 0))
    h1 = avg_pool(g, h0)
    check_close(h1, F.mean(h0, 0))
    h1 = max_pool(g, h0)
    check_close(h1, F.max(h0, 0))
    h1 = sort_pool(g, h0)
    assert h1.shape[0] == 10 * 5 and h1.ndim == 1

    # test#2: batched graph
    g_ = dgl.DGLGraph(nx.path_graph(5))
    bg = dgl.batch([g, g_, g, g_, g])
    h0 = F.randn((bg.number_of_nodes(), 5))
    h1 = sum_pool(bg, h0)
    truth = mx.nd.stack(F.sum(h0[:15], 0),
                        F.sum(h0[15:20], 0),
                        F.sum(h0[20:35], 0),
                        F.sum(h0[35:40], 0),
                        F.sum(h0[40:55], 0), axis=0)
    check_close(h1, truth)

    h1 = avg_pool(bg, h0)
    truth = mx.nd.stack(F.mean(h0[:15], 0),
                        F.mean(h0[15:20], 0),
                        F.mean(h0[20:35], 0),
                        F.mean(h0[35:40], 0),
                        F.mean(h0[40:55], 0), axis=0)
    check_close(h1, truth)

    h1 = max_pool(bg, h0)
    truth = mx.nd.stack(F.max(h0[:15], 0),
                        F.max(h0[15:20], 0),
                        F.max(h0[20:35], 0),
                        F.max(h0[35:40], 0),
                        F.max(h0[40:55], 0), axis=0)
    check_close(h1, truth)

    h1 = sort_pool(bg, h0)
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

def test_partial_edge_softmax():
    g = dgl.DGLGraph()
    g.add_nodes(30)
    # build a complete graph
    for i in range(30):
        for j in range(30):
            g.add_edge(i, j)

    score = F.randn((300, 1))
    score.attach_grad()
    grad = F.randn((300, 1))
    import numpy as np
    eids = np.random.choice(900, 300, replace=False)
    eids = F.zerocopy_from_numpy(eids)
    # compute partial edge softmax
    with mx.autograd.record():
        y_1 = nn.edge_softmax(g, score, eids)
        y_1.backward(grad)
        grad_1 = score.grad

    # compute edge softmax on edge subgraph
    subg = g.edge_subgraph(eids)
    with mx.autograd.record():
        y_2 = nn.edge_softmax(subg, score)
        y_2.backward(grad)
        grad_2 = score.grad

    assert F.allclose(y_1, y_2)
    assert F.allclose(grad_1, grad_2)

def test_rgcn():
    ctx = F.ctx()
    etype = []
    g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
    # 5 etypes
    R = 5
    for i in range(g.number_of_edges()):
        etype.append(i % 5)
    B = 2
    I = 10
    O = 8

    rgc_basis = nn.RelGraphConv(I, O, R, "basis", B)
    rgc_basis.initialize(ctx=ctx)
    h = nd.random.randn(100, I, ctx=ctx)
    r = nd.array(etype, ctx=ctx)
    h_new = rgc_basis(g, h, r)
    assert list(h_new.shape) == [100, O]

    rgc_bdd = nn.RelGraphConv(I, O, R, "bdd", B)
    rgc_bdd.initialize(ctx=ctx)
    h = nd.random.randn(100, I, ctx=ctx)
    r = nd.array(etype, ctx=ctx)
    h_new = rgc_bdd(g, h, r)
    assert list(h_new.shape) == [100, O]

    # with norm
    norm = nd.zeros((g.number_of_edges(), 1), ctx=ctx)

    rgc_basis = nn.RelGraphConv(I, O, R, "basis", B)
    rgc_basis.initialize(ctx=ctx)
    h = nd.random.randn(100, I, ctx=ctx)
    r = nd.array(etype, ctx=ctx)
    h_new = rgc_basis(g, h, r, norm)
    assert list(h_new.shape) == [100, O]

    rgc_bdd = nn.RelGraphConv(I, O, R, "bdd", B)
    rgc_bdd.initialize(ctx=ctx)
    h = nd.random.randn(100, I, ctx=ctx)
    r = nd.array(etype, ctx=ctx)
    h_new = rgc_bdd(g, h, r, norm)
    assert list(h_new.shape) == [100, O]

    # id input
    rgc_basis = nn.RelGraphConv(I, O, R, "basis", B)
    rgc_basis.initialize(ctx=ctx)
    h = nd.random.randint(0, I, (100,), ctx=ctx)
    r = nd.array(etype, ctx=ctx)
    h_new = rgc_basis(g, h, r)
    assert list(h_new.shape) == [100, O]

if __name__ == '__main__':
    test_graph_conv()
    test_edge_softmax()
    test_partial_edge_softmax()
    test_set2set()
    test_glob_att_pool()
    test_simple_pool()
    test_rgcn()
