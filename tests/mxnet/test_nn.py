import mxnet as mx
import networkx as nx
import numpy as np
import scipy as sp
import pytest
import dgl
import dgl.nn.mxnet as nn
import dgl.function as fn
import backend as F
from test_utils.graph_cases import get_cases, random_graph, random_bipartite, random_dglgraph
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

    conv = nn.GraphConv(5, 2, norm='none', bias=True)
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

@pytest.mark.parametrize('g', get_cases(['path', 'bipartite', 'small'], exclude=['zero-degree']))
@pytest.mark.parametrize('norm', ['none', 'both', 'right'])
@pytest.mark.parametrize('weight', [True, False])
@pytest.mark.parametrize('bias', [False])
def test_graph_conv2(g, norm, weight, bias):
    conv = nn.GraphConv(5, 2, norm=norm, weight=weight, bias=bias)
    conv.initialize(ctx=F.ctx())
    ext_w = F.randn((5, 2)).as_in_context(F.ctx())
    nsrc = g.number_of_nodes() if isinstance(g, dgl.DGLGraph) else g.number_of_src_nodes()
    ndst = g.number_of_nodes() if isinstance(g, dgl.DGLGraph) else g.number_of_dst_nodes()
    h = F.randn((nsrc, 5)).as_in_context(F.ctx())
    if weight:
        h = conv(g, h)
    else:
        h = conv(g, h, ext_w)
    assert h.shape == (ndst, 2)

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

def test_gat_conv():
    ctx = F.ctx()

    g = dgl.DGLGraph(nx.erdos_renyi_graph(20, 0.3))
    gat = nn.GATConv(10, 20, 5) # n_heads = 5
    gat.initialize(ctx=ctx)
    print(gat)

    # test#1: basic
    feat = F.randn((20, 10))
    h = gat(g, feat)
    assert h.shape == (20, 5, 20)

    # test#2: bipartite
    g = dgl.bipartite(sp.sparse.random(100, 200, density=0.1))
    gat = nn.GATConv((5, 10), 2, 4)
    gat.initialize(ctx=ctx)
    feat = (F.randn((100, 5)), F.randn((200, 10)))
    h = gat(g, feat)
    assert h.shape == (200, 4, 2)


@pytest.mark.parametrize('aggre_type', ['mean', 'pool', 'gcn'])
def test_sage_conv(aggre_type):
    ctx = F.ctx()
    g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
    sage = nn.SAGEConv(5, 10, aggre_type)
    feat = F.randn((100, 5))
    sage.initialize(ctx=ctx)
    h = sage(g, feat)
    assert h.shape[-1] == 10

    g = dgl.graph(sp.sparse.random(100, 100, density=0.1))
    sage = nn.SAGEConv(5, 10, aggre_type)
    feat = F.randn((100, 5))
    sage.initialize(ctx=ctx)
    h = sage(g, feat)
    assert h.shape[-1] == 10

    g = dgl.bipartite(sp.sparse.random(100, 200, density=0.1))
    dst_dim = 5 if aggre_type != 'gcn' else 10
    sage = nn.SAGEConv((10, dst_dim), 2, aggre_type)
    feat = (F.randn((100, 10)), F.randn((200, dst_dim)))
    sage.initialize(ctx=ctx)
    h = sage(g, feat)
    assert h.shape[-1] == 2
    assert h.shape[0] == 200

def test_gg_conv():
    g = dgl.DGLGraph(nx.erdos_renyi_graph(20, 0.3))
    ctx = F.ctx()

    gg_conv = nn.GatedGraphConv(10, 20, 3, 4) # n_step = 3, n_etypes = 4
    gg_conv.initialize(ctx=ctx)
    print(gg_conv)

    # test#1: basic
    h0 = F.randn((20, 10))
    etypes = nd.random.randint(0, 4, g.number_of_edges()).as_in_context(ctx)
    h1 = gg_conv(g, h0, etypes)
    assert h1.shape == (20, 20)

def test_cheb_conv():
    g = dgl.DGLGraph(nx.erdos_renyi_graph(20, 0.3))
    ctx = F.ctx()

    cheb = nn.ChebConv(10, 20, 3) # k = 3
    cheb.initialize(ctx=ctx)
    print(cheb)

    # test#1: basic
    h0 = F.randn((20, 10))
    h1 = cheb(g, h0)
    assert h1.shape == (20, 20)

def test_agnn_conv():
    g = dgl.DGLGraph(nx.erdos_renyi_graph(20, 0.3))
    ctx = F.ctx()

    agnn_conv = nn.AGNNConv(0.1, True)
    agnn_conv.initialize(ctx=ctx)
    print(agnn_conv)

    # test#1: basic
    feat = F.randn((20, 10))
    h = agnn_conv(g, feat)
    assert h.shape == (20, 10)

    g = dgl.bipartite(sp.sparse.random(100, 200, density=0.1))
    feat = (F.randn((100, 5)), F.randn((200, 5)))
    h = agnn_conv(g, feat)
    assert h.shape == (200, 5)

def test_appnp_conv():
    g = dgl.DGLGraph(nx.erdos_renyi_graph(20, 0.3))
    ctx = F.ctx()

    appnp_conv = nn.APPNPConv(3, 0.1, 0)
    appnp_conv.initialize(ctx=ctx)
    print(appnp_conv)

    # test#1: basic
    h0 = F.randn((20, 10))
    h1 = appnp_conv(g, h0)
    assert h1.shape == (20, 10)

def test_dense_cheb_conv():
    for k in range(1, 4):
        ctx = F.ctx()
        g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.3), readonly=True)
        adj = g.adjacency_matrix(ctx=ctx).tostype('default')
        cheb = nn.ChebConv(5, 2, k)
        dense_cheb = nn.DenseChebConv(5, 2, k)
        cheb.initialize(ctx=ctx)
        dense_cheb.initialize(ctx=ctx)

        for i in range(len(cheb.fc)):
            dense_cheb.fc[i].weight.set_data(
                cheb.fc[i].weight.data())
            if cheb.bias is not None:
                dense_cheb.bias.set_data(
                    cheb.bias.data())

        feat = F.randn((100, 5))
        out_cheb = cheb(g, feat, [2.0])
        out_dense_cheb = dense_cheb(adj, feat, 2.0)
        assert F.allclose(out_cheb, out_dense_cheb)

@pytest.mark.parametrize('norm_type', ['both', 'right', 'none'])
@pytest.mark.parametrize('g', [random_graph(100), random_bipartite(100, 200)])
def test_dense_graph_conv(g, norm_type):
    ctx = F.ctx()
    adj = g.adjacency_matrix(ctx=ctx).tostype('default')
    conv = nn.GraphConv(5, 2, norm=norm_type, bias=True)
    dense_conv = nn.DenseGraphConv(5, 2, norm=norm_type, bias=True)
    conv.initialize(ctx=ctx)
    dense_conv.initialize(ctx=ctx)
    dense_conv.weight.set_data(
        conv.weight.data())
    dense_conv.bias.set_data(
        conv.bias.data())
    feat = F.randn((g.number_of_src_nodes(), 5))
    out_conv = conv(g, feat)
    out_dense_conv = dense_conv(adj, feat)
    assert F.allclose(out_conv, out_dense_conv)

@pytest.mark.parametrize('g', [random_graph(100), random_bipartite(100, 200)])
def test_dense_sage_conv(g):
    ctx = F.ctx()
    adj = g.adjacency_matrix(ctx=ctx).tostype('default')
    sage = nn.SAGEConv(5, 2, 'gcn')
    dense_sage = nn.DenseSAGEConv(5, 2)
    sage.initialize(ctx=ctx)
    dense_sage.initialize(ctx=ctx)
    dense_sage.fc.weight.set_data(
        sage.fc_neigh.weight.data())
    dense_sage.fc.bias.set_data(
        sage.fc_neigh.bias.data())
    if len(g.ntypes) == 2:
        feat = (
            F.randn((g.number_of_src_nodes(), 5)),
            F.randn((g.number_of_dst_nodes(), 5))
        )
    else:
        feat = F.randn((g.number_of_nodes(), 5))

    out_sage = sage(g, feat)
    out_dense_sage = dense_sage(adj, feat)
    assert F.allclose(out_sage, out_dense_sage)

@pytest.mark.parametrize('g', [random_dglgraph(20), random_graph(20), random_bipartite(20, 10)])
def test_edge_conv(g):
    ctx = F.ctx()

    edge_conv = nn.EdgeConv(5, 2)
    edge_conv.initialize(ctx=ctx)
    print(edge_conv)

    # test #1: basic
    h0 = F.randn((g.number_of_src_nodes(), 5))
    if not g.is_homograph():
        # bipartite
        h1 = edge_conv(g, (h0, h0[:10]))
    else:
        h1 = edge_conv(g, h0)
    assert h1.shape == (g.number_of_dst_nodes(), 2)

def test_gin_conv():
    g = dgl.DGLGraph(nx.erdos_renyi_graph(20, 0.3))
    ctx = F.ctx()

    gin_conv = nn.GINConv(lambda x: x, 'mean', 0.1)
    gin_conv.initialize(ctx=ctx)
    print(gin_conv)

    # test #1: basic
    feat = F.randn((g.number_of_nodes(), 5))
    h = gin_conv(g, feat)
    assert h.shape == (20, 5)

    # test #2: bipartite
    g = dgl.bipartite(sp.sparse.random(100, 200, density=0.1))
    feat = (F.randn((100, 5)), F.randn((200, 5)))
    h = gin_conv(g, feat)
    return h.shape == (20, 5)


def test_gmm_conv():
    ctx = F.ctx()

    g = dgl.DGLGraph(nx.erdos_renyi_graph(20, 0.3))
    gmm_conv = nn.GMMConv(5, 2, 5, 3, 'max')
    gmm_conv.initialize(ctx=ctx)
    # test #1: basic
    h0 = F.randn((g.number_of_nodes(), 5))
    pseudo = F.randn((g.number_of_edges(), 5))
    h1 = gmm_conv(g, h0, pseudo)
    assert h1.shape == (g.number_of_nodes(), 2)

    g = dgl.graph(nx.erdos_renyi_graph(20, 0.3))
    gmm_conv = nn.GMMConv(5, 2, 5, 3, 'max')
    gmm_conv.initialize(ctx=ctx)
    # test #1: basic
    h0 = F.randn((g.number_of_nodes(), 5))
    pseudo = F.randn((g.number_of_edges(), 5))
    h1 = gmm_conv(g, h0, pseudo)
    assert h1.shape == (g.number_of_nodes(), 2)

    g = dgl.bipartite(sp.sparse.random(20, 10, 0.1))
    gmm_conv = nn.GMMConv((5, 4), 2, 5, 3, 'max')
    gmm_conv.initialize(ctx=ctx)
    # test #1: basic
    h0 = F.randn((g.number_of_src_nodes(), 5))
    hd = F.randn((g.number_of_dst_nodes(), 4))
    pseudo = F.randn((g.number_of_edges(), 5))
    h1 = gmm_conv(g, (h0, hd), pseudo)
    assert h1.shape == (g.number_of_dst_nodes(), 2)

def test_nn_conv():
    ctx = F.ctx()

    g = dgl.DGLGraph(nx.erdos_renyi_graph(20, 0.3))
    nn_conv = nn.NNConv(5, 2, gluon.nn.Embedding(3, 5 * 2), 'max')
    nn_conv.initialize(ctx=ctx)
    # test #1: basic
    h0 = F.randn((g.number_of_nodes(), 5))
    etypes = nd.random.randint(0, 4, g.number_of_edges()).as_in_context(ctx)
    h1 = nn_conv(g, h0, etypes)
    assert h1.shape == (g.number_of_nodes(), 2)

    g = dgl.graph(nx.erdos_renyi_graph(20, 0.3))
    nn_conv = nn.NNConv(5, 2, gluon.nn.Embedding(3, 5 * 2), 'max')
    nn_conv.initialize(ctx=ctx)
    # test #1: basic
    h0 = F.randn((g.number_of_nodes(), 5))
    etypes = nd.random.randint(0, 4, g.number_of_edges()).as_in_context(ctx)
    h1 = nn_conv(g, h0, etypes)
    assert h1.shape == (g.number_of_nodes(), 2)

    g = dgl.bipartite(sp.sparse.random(20, 10, 0.3))
    nn_conv = nn.NNConv((5, 4), 2, gluon.nn.Embedding(3, 5 * 2), 'max')
    nn_conv.initialize(ctx=ctx)
    # test #1: basic
    h0 = F.randn((g.number_of_src_nodes(), 5))
    hd = F.randn((g.number_of_dst_nodes(), 4))
    etypes = nd.random.randint(0, 4, g.number_of_edges()).as_in_context(ctx)
    h1 = nn_conv(g, (h0, hd), etypes)
    assert h1.shape == (g.number_of_dst_nodes(), 2)

def test_sg_conv():
    g = dgl.DGLGraph(nx.erdos_renyi_graph(20, 0.3))
    ctx = F.ctx()

    sgc = nn.SGConv(5, 2, 2)
    sgc.initialize(ctx=ctx)
    print(sgc)

    # test #1: basic
    h0 = F.randn((g.number_of_nodes(), 5))
    h1 = sgc(g, h0)
    assert h1.shape == (g.number_of_nodes(), 2)

def test_set2set():
    g = dgl.DGLGraph(nx.path_graph(10))
    ctx = F.ctx()

    s2s = nn.Set2Set(5, 3, 3) # hidden size 5, 3 iters, 3 layers
    s2s.initialize(ctx=ctx)
    print(s2s)

    # test#1: basic
    h0 = F.randn((g.number_of_nodes(), 5))
    h1 = s2s(g, h0)
    assert h1.shape[0] == 1 and h1.shape[1] == 10 and h1.ndim == 2

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
    assert h1.shape[0] == 1 and h1.shape[1] == 10 and h1.ndim == 2

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
    check_close(F.squeeze(h1, 0), F.sum(h0, 0))
    h1 = avg_pool(g, h0)
    check_close(F.squeeze(h1, 0), F.mean(h0, 0))
    h1 = max_pool(g, h0)
    check_close(F.squeeze(h1, 0), F.max(h0, 0))
    h1 = sort_pool(g, h0)
    assert h1.shape[0] == 1 and h1.shape[1] == 10 * 5 and h1.ndim == 2

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
    eids = np.random.choice(900, 300, replace=False).astype('int64')
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

def test_sequential():
    ctx = F.ctx()
    # test single graph
    class ExampleLayer(gluon.nn.Block):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def forward(self, graph, n_feat, e_feat):
            graph = graph.local_var()
            graph.ndata['h'] = n_feat
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            n_feat += graph.ndata['h']
            graph.apply_edges(fn.u_add_v('h', 'h', 'e'))
            e_feat += graph.edata['e']
            return n_feat, e_feat

    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 1, 2, 0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1, 2, 2, 2])
    net = nn.Sequential()
    net.add(ExampleLayer())
    net.add(ExampleLayer())
    net.add(ExampleLayer())
    net.initialize(ctx=ctx)
    n_feat = F.randn((3, 4))
    e_feat = F.randn((9, 4))
    n_feat, e_feat = net(g, n_feat, e_feat)
    assert n_feat.shape == (3, 4)
    assert e_feat.shape == (9, 4)

    # test multiple graphs
    class ExampleLayer(gluon.nn.Block):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def forward(self, graph, n_feat):
            graph = graph.local_var()
            graph.ndata['h'] = n_feat
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            n_feat += graph.ndata['h']
            return n_feat.reshape(graph.number_of_nodes() // 2, 2, -1).sum(1)

    g1 = dgl.DGLGraph(nx.erdos_renyi_graph(32, 0.05))
    g2 = dgl.DGLGraph(nx.erdos_renyi_graph(16, 0.2))
    g3 = dgl.DGLGraph(nx.erdos_renyi_graph(8, 0.8))
    net = nn.Sequential()
    net.add(ExampleLayer())
    net.add(ExampleLayer())
    net.add(ExampleLayer())
    net.initialize(ctx=ctx)
    n_feat = F.randn((32, 4))
    n_feat = net([g1, g2, g3], n_feat)
    assert n_feat.shape == (4, 4)

def myagg(alist, dsttype):
    rst = alist[0]
    for i in range(1, len(alist)):
        rst = rst + (i + 1) * alist[i]
    return rst

@pytest.mark.parametrize('agg', ['sum', 'max', 'min', 'mean', 'stack', myagg])
def test_hetero_conv(agg):
    g = dgl.heterograph({
        ('user', 'follows', 'user'): [(0, 1), (0, 2), (2, 1), (1, 3)],
        ('user', 'plays', 'game'): [(0, 0), (0, 2), (0, 3), (1, 0), (2, 2)],
        ('store', 'sells', 'game'): [(0, 0), (0, 3), (1, 1), (1, 2)]})
    conv = nn.HeteroGraphConv({
        'follows': nn.GraphConv(2, 3),
        'plays': nn.GraphConv(2, 4),
        'sells': nn.GraphConv(3, 4)},
        agg)
    conv.initialize(ctx=F.ctx())
    print(conv)
    uf = F.randn((4, 2))
    gf = F.randn((4, 4))
    sf = F.randn((2, 3))
    uf_dst = F.randn((4, 3))
    gf_dst = F.randn((4, 4))

    h = conv(g, {'user': uf})
    assert set(h.keys()) == {'user', 'game'}
    if agg != 'stack':
        assert h['user'].shape == (4, 3)
        assert h['game'].shape == (4, 4)
    else:
        assert h['user'].shape == (4, 1, 3)
        assert h['game'].shape == (4, 1, 4)

    h = conv(g, {'user': uf, 'store': sf})
    assert set(h.keys()) == {'user', 'game'}
    if agg != 'stack':
        assert h['user'].shape == (4, 3)
        assert h['game'].shape == (4, 4)
    else:
        assert h['user'].shape == (4, 1, 3)
        assert h['game'].shape == (4, 2, 4)

    h = conv(g, {'store': sf})
    assert set(h.keys()) == {'game'}
    if agg != 'stack':
        assert h['game'].shape == (4, 4)
    else:
        assert h['game'].shape == (4, 1, 4)

    # test with pair input
    conv = nn.HeteroGraphConv({
        'follows': nn.SAGEConv(2, 3, 'mean'),
        'plays': nn.SAGEConv((2, 4), 4, 'mean'),
        'sells': nn.SAGEConv(3, 4, 'mean')},
        agg)
    conv.initialize(ctx=F.ctx())

    h = conv(g, ({'user': uf}, {'user' : uf, 'game' : gf}))
    assert set(h.keys()) == {'user', 'game'}
    if agg != 'stack':
        assert h['user'].shape == (4, 3)
        assert h['game'].shape == (4, 4)
    else:
        assert h['user'].shape == (4, 1, 3)
        assert h['game'].shape == (4, 1, 4)

    # pair input requires both src and dst type features to be provided
    h = conv(g, ({'user': uf}, {'game' : gf}))
    assert set(h.keys()) == {'game'}
    if agg != 'stack':
        assert h['game'].shape == (4, 4)
    else:
        assert h['game'].shape == (4, 1, 4)

    # test with mod args
    class MyMod(mx.gluon.nn.Block):
        def __init__(self, s1, s2):
            super(MyMod, self).__init__()
            self.carg1 = 0
            self.s1 = s1
            self.s2 = s2
        def forward(self, g, h, arg1=None):  # mxnet does not support kwargs
            if arg1 is not None:
                self.carg1 += 1
            return F.zeros((g.number_of_dst_nodes(), self.s2))
    mod1 = MyMod(2, 3)
    mod2 = MyMod(2, 4)
    mod3 = MyMod(3, 4)
    conv = nn.HeteroGraphConv({
        'follows': mod1,
        'plays': mod2,
        'sells': mod3},
        agg)
    conv.initialize(ctx=F.ctx())
    mod_args = {'follows' : (1,), 'plays' : (1,)}
    h = conv(g, {'user' : uf, 'store' : sf}, mod_args)
    assert mod1.carg1 == 1
    assert mod2.carg1 == 1
    assert mod3.carg1 == 0

if __name__ == '__main__':
    test_graph_conv()
    test_gat_conv()
    test_sage_conv()
    test_gg_conv()
    test_cheb_conv()
    test_agnn_conv()
    test_appnp_conv()
    test_dense_cheb_conv()
    test_dense_graph_conv()
    test_dense_sage_conv()
    test_edge_conv()
    test_gin_conv()
    test_gmm_conv()
    test_nn_conv()
    test_sg_conv()
    test_edge_softmax()
    test_partial_edge_softmax()
    test_set2set()
    test_glob_att_pool()
    test_simple_pool()
    test_rgcn()
    test_sequential()
