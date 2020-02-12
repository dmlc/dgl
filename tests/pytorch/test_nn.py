import torch as th
import networkx as nx
import dgl
import dgl.nn.pytorch as nn
import dgl.function as fn
import backend as F
from copy import deepcopy

import numpy as np
import scipy as sp

def _AXWb(A, X, W, b):
    X = th.matmul(X, W)
    Y = th.matmul(A, X.view(X.shape[0], -1)).view_as(X)
    return Y + b

def test_graph_conv():
    g = dgl.DGLGraph(nx.path_graph(3))
    ctx = F.ctx()
    adj = g.adjacency_matrix(ctx=ctx)

    conv = nn.GraphConv(5, 2, norm=False, bias=True)
    conv = conv.to(ctx)
    print(conv)
    # test#1: basic
    h0 = F.ones((3, 5))
    h1 = conv(g, h0)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    assert F.allclose(h1, _AXWb(adj, h0, conv.weight, conv.bias))
    # test#2: more-dim
    h0 = F.ones((3, 5, 5))
    h1 = conv(g, h0)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    assert F.allclose(h1, _AXWb(adj, h0, conv.weight, conv.bias))

    conv = nn.GraphConv(5, 2)
    conv = conv.to(ctx)
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
    conv = conv.to(ctx)
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

    # test rest_parameters
    old_weight = deepcopy(conv.weight.data)
    conv.reset_parameters()
    new_weight = conv.weight.data
    assert not F.allclose(old_weight, new_weight)

def _S2AXWb(A, N, X, W, b):
    X1 = X * N
    X1 = th.matmul(A, X1.view(X1.shape[0], -1))
    X1 = X1 * N
    X2 = X1 * N
    X2 = th.matmul(A, X2.view(X2.shape[0], -1))
    X2 = X2 * N
    X = th.cat([X, X1, X2], dim=-1)
    Y = th.matmul(X, W.rot90())

    return Y + b

def test_tagconv():
    g = dgl.DGLGraph(nx.path_graph(3))
    ctx = F.ctx()
    adj = g.adjacency_matrix(ctx=ctx)
    norm = th.pow(g.in_degrees().float(), -0.5)

    conv = nn.TAGConv(5, 2, bias=True)
    conv = conv.to(ctx)
    print(conv)

    # test#1: basic
    h0 = F.ones((3, 5))
    h1 = conv(g, h0)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    shp = norm.shape + (1,) * (h0.dim() - 1)
    norm = th.reshape(norm, shp).to(ctx)

    assert F.allclose(h1, _S2AXWb(adj, norm, h0, conv.lin.weight, conv.lin.bias))

    conv = nn.TAGConv(5, 2)
    conv = conv.to(ctx)

    # test#2: basic
    h0 = F.ones((3, 5))
    h1 = conv(g, h0)
    assert h1.shape[-1] == 2

    # test reset_parameters
    old_weight = deepcopy(conv.lin.weight.data)
    conv.reset_parameters()
    new_weight = conv.lin.weight.data
    assert not F.allclose(old_weight, new_weight)

def test_set2set():
    ctx = F.ctx()
    g = dgl.DGLGraph(nx.path_graph(10))

    s2s = nn.Set2Set(5, 3, 3) # hidden size 5, 3 iters, 3 layers
    s2s = s2s.to(ctx)
    print(s2s)

    # test#1: basic
    h0 = F.randn((g.number_of_nodes(), 5))
    h1 = s2s(g, h0)
    assert h1.shape[0] == 10 and h1.dim() == 1

    # test#2: batched graph
    g1 = dgl.DGLGraph(nx.path_graph(11))
    g2 = dgl.DGLGraph(nx.path_graph(5))
    bg = dgl.batch([g, g1, g2])
    h0 = F.randn((bg.number_of_nodes(), 5))
    h1 = s2s(bg, h0)
    assert h1.shape[0] == 3 and h1.shape[1] == 10 and h1.dim() == 2

def test_glob_att_pool():
    ctx = F.ctx()
    g = dgl.DGLGraph(nx.path_graph(10))

    gap = nn.GlobalAttentionPooling(th.nn.Linear(5, 1), th.nn.Linear(5, 10))
    gap = gap.to(ctx)
    print(gap)

    # test#1: basic
    h0 = F.randn((g.number_of_nodes(), 5))
    h1 = gap(g, h0)
    assert h1.shape[0] == 10 and h1.dim() == 1

    # test#2: batched graph
    bg = dgl.batch([g, g, g, g])
    h0 = F.randn((bg.number_of_nodes(), 5))
    h1 = gap(bg, h0)
    assert h1.shape[0] == 4 and h1.shape[1] == 10 and h1.dim() == 2

def test_simple_pool():
    ctx = F.ctx()
    g = dgl.DGLGraph(nx.path_graph(15))

    sum_pool = nn.SumPooling()
    avg_pool = nn.AvgPooling()
    max_pool = nn.MaxPooling()
    sort_pool = nn.SortPooling(10) # k = 10
    print(sum_pool, avg_pool, max_pool, sort_pool)

    # test#1: basic
    h0 = F.randn((g.number_of_nodes(), 5))
    sum_pool = sum_pool.to(ctx)
    avg_pool = avg_pool.to(ctx)
    max_pool = max_pool.to(ctx)
    sort_pool = sort_pool.to(ctx)
    h1 = sum_pool(g, h0)
    assert F.allclose(h1, F.sum(h0, 0))
    h1 = avg_pool(g, h0)
    assert F.allclose(h1, F.mean(h0, 0))
    h1 = max_pool(g, h0)
    assert F.allclose(h1, F.max(h0, 0))
    h1 = sort_pool(g, h0)
    assert h1.shape[0] == 10 * 5 and h1.dim() == 1

    # test#2: batched graph
    g_ = dgl.DGLGraph(nx.path_graph(5))
    bg = dgl.batch([g, g_, g, g_, g])
    h0 = F.randn((bg.number_of_nodes(), 5))
    h1 = sum_pool(bg, h0)
    truth = th.stack([F.sum(h0[:15], 0),
                      F.sum(h0[15:20], 0),
                      F.sum(h0[20:35], 0),
                      F.sum(h0[35:40], 0),
                      F.sum(h0[40:55], 0)], 0)
    assert F.allclose(h1, truth)

    h1 = avg_pool(bg, h0)
    truth = th.stack([F.mean(h0[:15], 0),
                      F.mean(h0[15:20], 0),
                      F.mean(h0[20:35], 0),
                      F.mean(h0[35:40], 0),
                      F.mean(h0[40:55], 0)], 0)
    assert F.allclose(h1, truth)

    h1 = max_pool(bg, h0)
    truth = th.stack([F.max(h0[:15], 0),
                      F.max(h0[15:20], 0),
                      F.max(h0[20:35], 0),
                      F.max(h0[35:40], 0),
                      F.max(h0[40:55], 0)], 0)
    assert F.allclose(h1, truth)

    h1 = sort_pool(bg, h0)
    assert h1.shape[0] == 5 and h1.shape[1] == 10 * 5 and h1.dim() == 2

def test_set_trans():
    ctx = F.ctx()
    g = dgl.DGLGraph(nx.path_graph(15))

    st_enc_0 = nn.SetTransformerEncoder(50, 5, 10, 100, 2, 'sab')
    st_enc_1 = nn.SetTransformerEncoder(50, 5, 10, 100, 2, 'isab', 3)
    st_dec = nn.SetTransformerDecoder(50, 5, 10, 100, 2, 4)
    st_enc_0 = st_enc_0.to(ctx)
    st_enc_1 = st_enc_1.to(ctx)
    st_dec = st_dec.to(ctx)
    print(st_enc_0, st_enc_1, st_dec)

    # test#1: basic
    h0 = F.randn((g.number_of_nodes(), 50))
    h1 = st_enc_0(g, h0)
    assert h1.shape == h0.shape
    h1 = st_enc_1(g, h0)
    assert h1.shape == h0.shape
    h2 = st_dec(g, h1)
    assert h2.shape[0] == 200 and h2.dim() == 1

    # test#2: batched graph
    g1 = dgl.DGLGraph(nx.path_graph(5))
    g2 = dgl.DGLGraph(nx.path_graph(10))
    bg = dgl.batch([g, g1, g2])
    h0 = F.randn((bg.number_of_nodes(), 50))
    h1 = st_enc_0(bg, h0)
    assert h1.shape == h0.shape
    h1 = st_enc_1(bg, h0)
    assert h1.shape == h0.shape

    h2 = st_dec(bg, h1)
    assert h2.shape[0] == 3 and h2.shape[1] == 200 and h2.dim() == 2

def uniform_attention(g, shape):
    a = th.ones(shape)
    target_shape = (g.number_of_edges(),) + (1,) * (len(shape) - 1)
    return a / g.in_degrees(g.edges()[1]).view(target_shape).float()

def test_edge_softmax():
    # Basic
    g = dgl.DGLGraph(nx.path_graph(3))
    edata = F.ones((g.number_of_edges(), 1))
    a = nn.edge_softmax(g, edata)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    assert F.allclose(a, uniform_attention(g, a.shape))

    # Test higher dimension case
    edata = F.ones((g.number_of_edges(), 3, 1))
    a = nn.edge_softmax(g, edata)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    assert F.allclose(a, uniform_attention(g, a.shape))

    # Test both forward and backward with PyTorch built-in softmax.
    g = dgl.DGLGraph()
    g.add_nodes(30)
    # build a complete graph
    for i in range(30):
        for j in range(30):
            g.add_edge(i, j)

    score = F.randn((900, 1))
    score.requires_grad_()
    grad = F.randn((900, 1))
    y = F.softmax(score.view(30, 30), dim=0).view(-1, 1)
    y.backward(grad)
    grad_score = score.grad
    score.grad.zero_()
    y_dgl = nn.edge_softmax(g, score)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    # check forward
    assert F.allclose(y_dgl, y)
    y_dgl.backward(grad)
    # checkout gradient
    assert F.allclose(score.grad, grad_score)
    print(score.grad[:10], grad_score[:10])
    
    # Test 2
    def generate_rand_graph(n):
      arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
      return dgl.DGLGraph(arr, readonly=True)
    
    g = generate_rand_graph(50)
    a1 = F.randn((g.number_of_edges(), 1)).requires_grad_()
    a2 = a1.clone().detach().requires_grad_()
    g.edata['s'] = a1
    g.group_apply_edges('dst', lambda edges: {'ss':F.softmax(edges.data['s'], 1)})
    g.edata['ss'].sum().backward()
    
    builtin_sm = nn.edge_softmax(g, a2)
    builtin_sm.sum().backward()
    print(a1.grad - a2.grad)
    assert len(g.ndata) == 0
    assert len(g.edata) == 2
    assert F.allclose(a1.grad, a2.grad, rtol=1e-4, atol=1e-4) # Follow tolerance in unittest backend

def test_partial_edge_softmax():
    g = dgl.DGLGraph()
    g.add_nodes(30)
    # build a complete graph
    for i in range(30):
        for j in range(30):
            g.add_edge(i, j)

    score = F.randn((300, 1))
    score.requires_grad_()
    grad = F.randn((300, 1))
    import numpy as np
    eids = np.random.choice(900, 300, replace=False).astype('int64')
    eids = F.zerocopy_from_numpy(eids)
    # compute partial edge softmax
    y_1 = nn.edge_softmax(g, score, eids)
    y_1.backward(grad)
    grad_1 = score.grad
    score.grad.zero_()
    # compute edge softmax on edge subgraph
    subg = g.edge_subgraph(eids)
    y_2 = nn.edge_softmax(subg, score)
    y_2.backward(grad)
    grad_2 = score.grad
    score.grad.zero_()

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

    rgc_basis = nn.RelGraphConv(I, O, R, "basis", B).to(ctx)
    h = th.randn((100, I)).to(ctx)
    r = th.tensor(etype).to(ctx)
    h_new = rgc_basis(g, h, r)
    assert list(h_new.shape) == [100, O]

    rgc_bdd = nn.RelGraphConv(I, O, R, "bdd", B).to(ctx)
    h = th.randn((100, I)).to(ctx)
    r = th.tensor(etype).to(ctx)
    h_new = rgc_bdd(g, h, r)
    assert list(h_new.shape) == [100, O]

    # with norm
    norm = th.zeros((g.number_of_edges(), 1)).to(ctx)

    rgc_basis = nn.RelGraphConv(I, O, R, "basis", B).to(ctx)
    h = th.randn((100, I)).to(ctx)
    r = th.tensor(etype).to(ctx)
    h_new = rgc_basis(g, h, r, norm)
    assert list(h_new.shape) == [100, O]

    rgc_bdd = nn.RelGraphConv(I, O, R, "bdd", B).to(ctx)
    h = th.randn((100, I)).to(ctx)
    r = th.tensor(etype).to(ctx)
    h_new = rgc_bdd(g, h, r, norm)
    assert list(h_new.shape) == [100, O]

    # id input
    rgc_basis = nn.RelGraphConv(I, O, R, "basis", B).to(ctx)
    h = th.randint(0, I, (100,)).to(ctx)
    r = th.tensor(etype).to(ctx)
    h_new = rgc_basis(g, h, r)
    assert list(h_new.shape) == [100, O]

def test_gat_conv():
    ctx = F.ctx()
    g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
    gat = nn.GATConv(5, 2, 4)
    feat = F.randn((100, 5))
    gat = gat.to(ctx)
    h = gat(g, feat)
    assert h.shape[-1] == 2 and h.shape[-2] == 4

def test_sage_conv():
    for aggre_type in ['mean', 'pool', 'gcn', 'lstm']:
        ctx = F.ctx()
        g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
        sage = nn.SAGEConv(5, 10, aggre_type)
        feat = F.randn((100, 5))
        sage = sage.to(ctx)
        h = sage(g, feat)
        assert h.shape[-1] == 10

def test_sgc_conv():
    ctx = F.ctx()
    g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
    # not cached
    sgc = nn.SGConv(5, 10, 3)
    feat = F.randn((100, 5))
    sgc = sgc.to(ctx)

    h = sgc(g, feat)
    assert h.shape[-1] == 10

    # cached
    sgc = nn.SGConv(5, 10, 3, True)
    sgc = sgc.to(ctx)
    h_0 = sgc(g, feat)
    h_1 = sgc(g, feat + 1)
    assert F.allclose(h_0, h_1)
    assert h_0.shape[-1] == 10

def test_appnp_conv():
    ctx = F.ctx()
    g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
    appnp = nn.APPNPConv(10, 0.1)
    feat = F.randn((100, 5))
    appnp = appnp.to(ctx)

    h = appnp(g, feat)
    assert h.shape[-1] == 5

def test_gin_conv():
    for aggregator_type in ['mean', 'max', 'sum']:
        ctx = F.ctx()
        g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
        gin = nn.GINConv(
            th.nn.Linear(5, 12),
            aggregator_type
        )
        feat = F.randn((100, 5))
        gin = gin.to(ctx)
        h = gin(g, feat)
        assert h.shape[-1] == 12

def test_agnn_conv():
    ctx = F.ctx()
    g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
    agnn = nn.AGNNConv(1)
    feat = F.randn((100, 5))
    agnn = agnn.to(ctx)
    h = agnn(g, feat)
    assert h.shape[-1] == 5

def test_gated_graph_conv():
    ctx = F.ctx()
    g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
    ggconv = nn.GatedGraphConv(5, 10, 5, 3)
    etypes = th.arange(g.number_of_edges()) % 3
    feat = F.randn((100, 5))
    ggconv = ggconv.to(ctx)
    etypes = etypes.to(ctx)

    h = ggconv(g, feat, etypes)
    # current we only do shape check
    assert h.shape[-1] == 10

def test_nn_conv():
    ctx = F.ctx()
    g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
    edge_func = th.nn.Linear(4, 5 * 10)
    nnconv = nn.NNConv(5, 10, edge_func, 'mean')
    feat = F.randn((100, 5))
    efeat = F.randn((g.number_of_edges(), 4))
    nnconv = nnconv.to(ctx)
    h = nnconv(g, feat, efeat)
    # currently we only do shape check
    assert h.shape[-1] == 10

def test_gmm_conv():
    ctx = F.ctx()
    g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
    gmmconv = nn.GMMConv(5, 10, 3, 4, 'mean')
    feat = F.randn((100, 5))
    pseudo = F.randn((g.number_of_edges(), 3))
    gmmconv = gmmconv.to(ctx)
    h = gmmconv(g, feat, pseudo)
    # currently we only do shape check
    assert h.shape[-1] == 10

def test_dense_graph_conv():
    ctx = F.ctx()
    g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
    adj = g.adjacency_matrix(ctx=ctx).to_dense()
    conv = nn.GraphConv(5, 2, norm=False, bias=True)
    dense_conv = nn.DenseGraphConv(5, 2, norm=False, bias=True)
    dense_conv.weight.data = conv.weight.data
    dense_conv.bias.data = conv.bias.data
    feat = F.randn((100, 5))
    conv = conv.to(ctx)
    dense_conv = dense_conv.to(ctx)
    out_conv = conv(g, feat)
    out_dense_conv = dense_conv(adj, feat)
    assert F.allclose(out_conv, out_dense_conv)

def test_dense_sage_conv():
    ctx = F.ctx()
    g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
    adj = g.adjacency_matrix(ctx=ctx).to_dense()
    sage = nn.SAGEConv(5, 2, 'gcn')
    dense_sage = nn.DenseSAGEConv(5, 2)
    dense_sage.fc.weight.data = sage.fc_neigh.weight.data
    dense_sage.fc.bias.data = sage.fc_neigh.bias.data
    feat = F.randn((100, 5))
    sage = sage.to(ctx)
    dense_sage = dense_sage.to(ctx)
    out_sage = sage(g, feat)
    out_dense_sage = dense_sage(adj, feat)
    assert F.allclose(out_sage, out_dense_sage)

def test_dense_cheb_conv():
    for k in range(1, 4):
        ctx = F.ctx()
        g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
        adj = g.adjacency_matrix(ctx=ctx).to_dense()
        cheb = nn.ChebConv(5, 2, k)
        dense_cheb = nn.DenseChebConv(5, 2, k)
        for i in range(len(cheb.fc)):
            dense_cheb.W.data[i] = cheb.fc[i].weight.data.t()
        if cheb.bias is not None:
            dense_cheb.bias.data = cheb.bias.data
        feat = F.randn((100, 5))
        cheb = cheb.to(ctx)
        dense_cheb = dense_cheb.to(ctx)
        out_cheb = cheb(g, feat, [2.0])
        out_dense_cheb = dense_cheb(adj, feat, 2.0)
        assert F.allclose(out_cheb, out_dense_cheb)

def test_sequential():
    ctx = F.ctx()
    # Test single graph
    class ExampleLayer(th.nn.Module):
        def __init__(self):
            super().__init__()

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
    net = nn.Sequential(ExampleLayer(), ExampleLayer(), ExampleLayer())
    n_feat = F.randn((3, 4))
    e_feat = F.randn((9, 4))
    net = net.to(ctx)
    n_feat, e_feat = net(g, n_feat, e_feat)
    assert n_feat.shape == (3, 4)
    assert e_feat.shape == (9, 4)

    # Test multiple graph
    class ExampleLayer(th.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, graph, n_feat):
            graph = graph.local_var()
            graph.ndata['h'] = n_feat
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            n_feat += graph.ndata['h']
            return n_feat.view(graph.number_of_nodes() // 2, 2, -1).sum(1)

    g1 = dgl.DGLGraph(nx.erdos_renyi_graph(32, 0.05))
    g2 = dgl.DGLGraph(nx.erdos_renyi_graph(16, 0.2))
    g3 = dgl.DGLGraph(nx.erdos_renyi_graph(8, 0.8))
    net = nn.Sequential(ExampleLayer(), ExampleLayer(), ExampleLayer())
    net = net.to(ctx)
    n_feat = F.randn((32, 4))
    n_feat = net([g1, g2, g3], n_feat)
    assert n_feat.shape == (4, 4)

def test_atomic_conv():
    g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
    aconv = nn.AtomicConv(interaction_cutoffs=F.tensor([12.0, 12.0]),
                          rbf_kernel_means=F.tensor([0.0, 2.0]),
                          rbf_kernel_scaling=F.tensor([4.0, 4.0]),
                          features_to_use=F.tensor([6.0, 8.0]))

    ctx = F.ctx()
    if F.gpu_ctx():
        aconv = aconv.to(ctx)

    feat = F.randn((100, 1))
    dist = F.randn((g.number_of_edges(), 1))

    h = aconv(g, feat, dist)
    # current we only do shape check
    assert h.shape[-1] == 4

def test_cf_conv():
    g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
    cfconv = nn.CFConv(node_in_feats=2,
                       edge_in_feats=3,
                       hidden_feats=2,
                       out_feats=3)

    ctx = F.ctx()
    if F.gpu_ctx():
        cfconv = cfconv.to(ctx)

    node_feats = F.randn((100, 2))
    edge_feats = F.randn((g.number_of_edges(), 3))
    h = cfconv(g, node_feats, edge_feats)
    # current we only do shape check
    assert h.shape[-1] == 3    

if __name__ == '__main__':
    test_graph_conv()
    test_edge_softmax()
    test_partial_edge_softmax()
    test_set2set()
    test_glob_att_pool()
    test_simple_pool()
    test_set_trans()
    test_rgcn()
    test_tagconv()
    test_gat_conv()
    test_sage_conv()
    test_sgc_conv()
    test_appnp_conv()
    test_gin_conv()
    test_agnn_conv()
    test_gated_graph_conv()
    test_nn_conv()
    test_gmm_conv()
    test_dense_graph_conv()
    test_dense_sage_conv()
    test_dense_cheb_conv()
    test_sequential()
    test_atomic_conv()
    test_cf_conv()
