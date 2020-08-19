import tensorflow as tf
from tensorflow.keras import layers
import networkx as nx
import pytest
import dgl
import dgl.nn.tensorflow as nn
import dgl.function as fn
import backend as F
from test_utils.graph_cases import get_cases, random_graph, random_bipartite, random_dglgraph
from test_utils import parametrize_dtype
from copy import deepcopy

import numpy as np
import scipy as sp

def _AXWb(A, X, W, b):
    X = tf.matmul(X, W)
    Y = tf.reshape(tf.matmul(A, tf.reshape(X, (X.shape[0], -1))), X.shape)
    return Y + b

def test_graph_conv():
    g = dgl.DGLGraph(nx.path_graph(3)).to(F.ctx())
    ctx = F.ctx()
    adj = tf.sparse.to_dense(tf.sparse.reorder(g.adjacency_matrix(transpose=False, ctx=ctx)))

    conv = nn.GraphConv(5, 2, norm='none', bias=True)
    # conv = conv
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
    # conv = conv
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
    # conv = conv
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
    # old_weight = deepcopy(conv.weight.data)
    # conv.reset_parameters()
    # new_weight = conv.weight.data
    # assert not F.allclose(old_weight, new_weight)

@parametrize_dtype
@pytest.mark.parametrize('g', get_cases(['homo', 'block-bipartite'], exclude=['zero-degree', 'dglgraph']))
@pytest.mark.parametrize('norm', ['none', 'both', 'right'])
@pytest.mark.parametrize('weight', [True, False])
@pytest.mark.parametrize('bias', [True, False])
def test_graph_conv2(idtype, g, norm, weight, bias):
    g = g.astype(idtype).to(F.ctx())
    conv = nn.GraphConv(5, 2, norm=norm, weight=weight, bias=bias)
    ext_w = F.randn((5, 2))
    nsrc = g.number_of_src_nodes()
    ndst = g.number_of_dst_nodes()
    h = F.randn((nsrc, 5))
    h_dst = F.randn((ndst, 2))
    if weight:
        h_out = conv(g, h)
    else:
        h_out = conv(g, h, weight=ext_w)
    assert h_out.shape == (ndst, 2)

@parametrize_dtype
@pytest.mark.parametrize('g', get_cases(['bipartite'], exclude=['zero-degree', 'dglgraph']))
@pytest.mark.parametrize('norm', ['none', 'both', 'right'])
@pytest.mark.parametrize('weight', [True, False])
@pytest.mark.parametrize('bias', [True, False])
def test_graph_conv2_bi(idtype, g, norm, weight, bias):
    g = g.astype(idtype).to(F.ctx())
    conv = nn.GraphConv(5, 2, norm=norm, weight=weight, bias=bias)
    ext_w = F.randn((5, 2))
    nsrc = g.number_of_src_nodes()
    ndst = g.number_of_dst_nodes()
    h = F.randn((nsrc, 5))
    h_dst = F.randn((ndst, 2))
    if weight:
        h_out = conv(g, (h, h_dst))
    else:
        h_out = conv(g, (h, h_dst), weight=ext_w)
    assert h_out.shape == (ndst, 2)

def test_simple_pool():
    ctx = F.ctx()
    g = dgl.DGLGraph(nx.path_graph(15)).to(F.ctx())

    sum_pool = nn.SumPooling()
    avg_pool = nn.AvgPooling()
    max_pool = nn.MaxPooling()
    sort_pool = nn.SortPooling(10) # k = 10
    print(sum_pool, avg_pool, max_pool, sort_pool)

    # test#1: basic
    h0 = F.randn((g.number_of_nodes(), 5))
    h1 = sum_pool(g, h0)
    assert F.allclose(F.squeeze(h1, 0), F.sum(h0, 0))
    h1 = avg_pool(g, h0)
    assert F.allclose(F.squeeze(h1, 0), F.mean(h0, 0))
    h1 = max_pool(g, h0)
    assert F.allclose(F.squeeze(h1, 0), F.max(h0, 0))
    h1 = sort_pool(g, h0)
    assert h1.shape[0] == 1 and h1.shape[1] == 10 * 5 and h1.ndim == 2

    # test#2: batched graph
    g_ = dgl.DGLGraph(nx.path_graph(5)).to(F.ctx())
    bg = dgl.batch([g, g_, g, g_, g])
    h0 = F.randn((bg.number_of_nodes(), 5))
    h1 = sum_pool(bg, h0)
    truth = tf.stack([F.sum(h0[:15], 0),
                      F.sum(h0[15:20], 0),
                      F.sum(h0[20:35], 0),
                      F.sum(h0[35:40], 0),
                      F.sum(h0[40:55], 0)], 0)
    assert F.allclose(h1, truth)

    h1 = avg_pool(bg, h0)
    truth = tf.stack([F.mean(h0[:15], 0),
                      F.mean(h0[15:20], 0),
                      F.mean(h0[20:35], 0),
                      F.mean(h0[35:40], 0),
                      F.mean(h0[40:55], 0)], 0)
    assert F.allclose(h1, truth)

    h1 = max_pool(bg, h0)
    truth = tf.stack([F.max(h0[:15], 0),
                      F.max(h0[15:20], 0),
                      F.max(h0[20:35], 0),
                      F.max(h0[35:40], 0),
                      F.max(h0[40:55], 0)], 0)
    assert F.allclose(h1, truth)

    h1 = sort_pool(bg, h0)
    assert h1.shape[0] == 5 and h1.shape[1] == 10 * 5 and h1.ndim == 2

def test_glob_att_pool():
    g = dgl.DGLGraph(nx.path_graph(10)).to(F.ctx())

    gap = nn.GlobalAttentionPooling(layers.Dense(1), layers.Dense(10))
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


def test_rgcn():
    etype = []
    g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True).to(F.ctx())
    # 5 etypes
    R = 5
    for i in range(g.number_of_edges()):
        etype.append(i % 5)
    B = 2
    I = 10
    O = 8

    rgc_basis = nn.RelGraphConv(I, O, R, "basis", B)
    rgc_basis_low = nn.RelGraphConv(I, O, R, "basis", B, low_mem=True)
    rgc_basis_low.weight = rgc_basis.weight
    rgc_basis_low.w_comp = rgc_basis.w_comp
    rgc_basis_low.loop_weight = rgc_basis.loop_weight
    h = tf.random.normal((100, I))
    r = tf.constant(etype)
    h_new = rgc_basis(g, h, r)
    h_new_low = rgc_basis_low(g, h, r)
    assert list(h_new.shape) == [100, O]
    assert list(h_new_low.shape) == [100, O]
    assert F.allclose(h_new, h_new_low)

    rgc_bdd = nn.RelGraphConv(I, O, R, "bdd", B)
    rgc_bdd_low = nn.RelGraphConv(I, O, R, "bdd", B, low_mem=True)
    rgc_bdd_low.weight = rgc_bdd.weight
    rgc_bdd_low.loop_weight = rgc_bdd.loop_weight
    h = tf.random.normal((100, I))
    r = tf.constant(etype)
    h_new = rgc_bdd(g, h, r)
    h_new_low = rgc_bdd_low(g, h, r)
    assert list(h_new.shape) == [100, O]
    assert list(h_new_low.shape) == [100, O]
    assert F.allclose(h_new, h_new_low)

    # with norm
    norm = tf.zeros((g.number_of_edges(), 1))

    rgc_basis = nn.RelGraphConv(I, O, R, "basis", B)
    rgc_basis_low = nn.RelGraphConv(I, O, R, "basis", B, low_mem=True)
    rgc_basis_low.weight = rgc_basis.weight
    rgc_basis_low.w_comp = rgc_basis.w_comp
    rgc_basis_low.loop_weight = rgc_basis.loop_weight
    h = tf.random.normal((100, I))
    r = tf.constant(etype)
    h_new = rgc_basis(g, h, r, norm)
    h_new_low = rgc_basis_low(g, h, r, norm)
    assert list(h_new.shape) == [100, O]
    assert list(h_new_low.shape) == [100, O]
    assert F.allclose(h_new, h_new_low)

    rgc_bdd = nn.RelGraphConv(I, O, R, "bdd", B)
    rgc_bdd_low = nn.RelGraphConv(I, O, R, "bdd", B, low_mem=True)
    rgc_bdd_low.weight = rgc_bdd.weight
    rgc_bdd_low.loop_weight = rgc_bdd.loop_weight
    h = tf.random.normal((100, I))
    r = tf.constant(etype)
    h_new = rgc_bdd(g, h, r, norm)
    h_new_low = rgc_bdd_low(g, h, r, norm)
    assert list(h_new.shape) == [100, O]
    assert list(h_new_low.shape) == [100, O]
    assert F.allclose(h_new, h_new_low)

    # id input
    rgc_basis = nn.RelGraphConv(I, O, R, "basis", B)
    rgc_basis_low = nn.RelGraphConv(I, O, R, "basis", B, low_mem=True)
    rgc_basis_low.weight = rgc_basis.weight
    rgc_basis_low.w_comp = rgc_basis.w_comp
    rgc_basis_low.loop_weight = rgc_basis.loop_weight
    h = tf.constant(np.random.randint(0, I, (100,))) * 1
    r = tf.constant(etype) * 1
    h_new = rgc_basis(g, h, r)
    h_new_low = rgc_basis_low(g, h, r)
    assert list(h_new.shape) == [100, O]
    assert list(h_new_low.shape) == [100, O]
    assert F.allclose(h_new, h_new_low)

@parametrize_dtype
@pytest.mark.parametrize('g', get_cases(['homo', 'block-bipartite'], exclude=['zero-degree']))
def test_gat_conv(g, idtype):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    gat = nn.GATConv(5, 2, 4)
    feat = F.randn((g.number_of_nodes(), 5))
    h = gat(g, feat)
    assert h.shape == (g.number_of_nodes(), 4, 2)

@parametrize_dtype
@pytest.mark.parametrize('g', get_cases(['bipartite'], exclude=['zero-degree']))
def test_gat_conv_bi(g, idtype):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    gat = nn.GATConv(5, 2, 4)
    feat = (F.randn((g.number_of_src_nodes(), 5)), F.randn((g.number_of_dst_nodes(), 5)))
    h = gat(g, feat)
    assert h.shape == (g.number_of_dst_nodes(), 4, 2)

@parametrize_dtype
@pytest.mark.parametrize('g', get_cases(['homo', 'block-bipartite']))
@pytest.mark.parametrize('aggre_type', ['mean', 'pool', 'gcn'])
def test_sage_conv(idtype, g, aggre_type):
    g = g.astype(idtype).to(F.ctx())
    sage = nn.SAGEConv(5, 10, aggre_type)
    feat = F.randn((g.number_of_nodes(), 5))
    h = sage(g, feat)
    assert h.shape[-1] == 10

@parametrize_dtype
@pytest.mark.parametrize('g', get_cases(['bipartite']))
@pytest.mark.parametrize('aggre_type', ['mean', 'pool', 'gcn'])
def test_sage_conv_bi(idtype, g, aggre_type):
    g = g.astype(idtype).to(F.ctx())
    sage = nn.SAGEConv(5, 10, aggre_type)
    dst_dim = 5 if aggre_type != 'gcn' else 10
    sage = nn.SAGEConv((10, dst_dim), 2, aggre_type)
    feat = (F.randn((g.number_of_src_nodes(), 10)), F.randn((g.number_of_dst_nodes(), dst_dim)))
    h = sage(g, feat)
    assert h.shape[-1] == 2
    assert h.shape[0] == g.number_of_dst_nodes()

@parametrize_dtype
@pytest.mark.parametrize('aggre_type', ['mean', 'pool', 'gcn'])
def test_sage_conv_bi_empty(idtype, aggre_type):
    # Test the case for graphs without edges
    g = dgl.heterograph({('_U', '_E', '_V'): ([], [])}, {'_U': 5, '_V': 3}).to(F.ctx())
    g = g.astype(idtype).to(F.ctx())
    sage = nn.SAGEConv((3, 3), 2, 'gcn')
    feat = (F.randn((5, 3)), F.randn((3, 3)))
    h = sage(g, feat)
    assert h.shape[-1] == 2
    assert h.shape[0] == 3
    for aggre_type in ['mean', 'pool', 'lstm']:
        sage = nn.SAGEConv((3, 1), 2, aggre_type)
        feat = (F.randn((5, 3)), F.randn((3, 1)))
        h = sage(g, feat)
        assert h.shape[-1] == 2
        assert h.shape[0] == 3

@parametrize_dtype
@pytest.mark.parametrize('g', get_cases(['homo'], exclude=['zero-degree']))
def test_sgc_conv(g, idtype):
    ctx = F.ctx()
    g = g.astype(idtype).to(ctx)
    # not cached
    sgc = nn.SGConv(5, 10, 3)
    feat = F.randn((g.number_of_nodes(), 5))

    h = sgc(g, feat)
    assert h.shape[-1] == 10

    # cached
    sgc = nn.SGConv(5, 10, 3, True)
    h_0 = sgc(g, feat)
    h_1 = sgc(g, feat + 1)
    assert F.allclose(h_0, h_1)
    assert h_0.shape[-1] == 10

@parametrize_dtype
@pytest.mark.parametrize('g', get_cases(['homo'], exclude=['zero-degree']))
def test_appnp_conv(g, idtype):
    ctx = F.ctx()
    g = g.astype(idtype).to(ctx)
    appnp = nn.APPNPConv(10, 0.1)
    feat = F.randn((g.number_of_nodes(), 5))

    h = appnp(g, feat)
    assert h.shape[-1] == 5

@parametrize_dtype
@pytest.mark.parametrize('g', get_cases(['homo', 'block-bipartite']))
@pytest.mark.parametrize('aggregator_type', ['mean', 'max', 'sum'])
def test_gin_conv(g, idtype, aggregator_type):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    gin = nn.GINConv(
        tf.keras.layers.Dense(12),
        aggregator_type
    )
    feat = F.randn((g.number_of_nodes(), 5))
    h = gin(g, feat)
    assert h.shape == (g.number_of_nodes(), 12)

@parametrize_dtype
@pytest.mark.parametrize('g', get_cases(['bipartite']))
@pytest.mark.parametrize('aggregator_type', ['mean', 'max', 'sum'])
def test_gin_conv_bi(g, idtype, aggregator_type):
    g = g.astype(idtype).to(F.ctx())
    gin = nn.GINConv(
        tf.keras.layers.Dense(12),
        aggregator_type
    )
    feat = (F.randn((g.number_of_src_nodes(), 5)), F.randn((g.number_of_dst_nodes(), 5)))
    h = gin(g, feat)
    assert h.shape == (g.number_of_dst_nodes(), 12)

def myagg(alist, dsttype):
    rst = alist[0]
    for i in range(1, len(alist)):
        rst = rst + (i + 1) * alist[i]
    return rst

@parametrize_dtype
@pytest.mark.parametrize('agg', ['sum', 'max', 'min', 'mean', 'stack', myagg])
def test_hetero_conv(agg, idtype):
    g = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 0, 2, 1], [1, 2, 1, 3]),
        ('user', 'plays', 'game'): ([0, 0, 0, 1, 2], [0, 2, 3, 0, 2]),
        ('store', 'sells', 'game'): ([0, 0, 1, 1], [0, 3, 1, 2])},
        idtype=idtype, device=F.ctx())
    conv = nn.HeteroGraphConv({
        'follows': nn.GraphConv(2, 3, allow_zero_in_degree=True),
        'plays': nn.GraphConv(2, 4, allow_zero_in_degree=True),
        'sells': nn.GraphConv(3, 4, allow_zero_in_degree=True)},
        agg)
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
    class MyMod(tf.keras.layers.Layer):
        def __init__(self, s1, s2):
            super(MyMod, self).__init__()
            self.carg1 = 0
            self.carg2 = 0
            self.s1 = s1
            self.s2 = s2
        def call(self, g, h, arg1=None, *, arg2=None):
            if arg1 is not None:
                self.carg1 += 1
            if arg2 is not None:
                self.carg2 += 1
            return tf.zeros((g.number_of_dst_nodes(), self.s2))
    mod1 = MyMod(2, 3)
    mod2 = MyMod(2, 4)
    mod3 = MyMod(3, 4)
    conv = nn.HeteroGraphConv({
        'follows': mod1,
        'plays': mod2,
        'sells': mod3},
        agg)
    mod_args = {'follows' : (1,), 'plays' : (1,)}
    mod_kwargs = {'sells' : {'arg2' : 'abc'}}
    h = conv(g, {'user' : uf, 'store' : sf}, mod_args=mod_args, mod_kwargs=mod_kwargs)
    assert mod1.carg1 == 1
    assert mod1.carg2 == 0
    assert mod2.carg1 == 1
    assert mod2.carg2 == 0
    assert mod3.carg1 == 0
    assert mod3.carg2 == 1


def test_dense_cheb_conv():
    for k in range(3, 4):
        ctx = F.ctx()
        g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1, random_state=42))
        g = g.to(ctx)

        adj = tf.sparse.to_dense(tf.sparse.reorder(g.adjacency_matrix(transpose=False, ctx=ctx)))
        cheb = nn.ChebConv(5, 2, k, None, bias=True)
        dense_cheb = nn.DenseChebConv(5, 2, k, bias=True)

        # init cheb modules
        feat = F.ones((100, 5))
        out_cheb = cheb(g, feat, [2.0])

        dense_cheb.W = tf.reshape(cheb.linear.weights[0], (k, 5, 2))
        if cheb.linear.bias is not None:
            dense_cheb.bias = cheb.linear.bias

        out_dense_cheb = dense_cheb(adj, feat, 2.0)
        print(out_cheb - out_dense_cheb)
        assert F.allclose(out_cheb, out_dense_cheb)


if __name__ == '__main__':
    test_graph_conv()
    # test_set2set()
    test_glob_att_pool()
    test_simple_pool()
    # test_set_trans()
    test_rgcn()
    # test_tagconv()
    test_gat_conv()
    test_sage_conv()
    test_sgc_conv()
    test_appnp_conv()
    test_gin_conv()
    # test_agnn_conv()
    # test_gated_graph_conv()
    # test_nn_conv()
    # test_gmm_conv()
    # test_dense_graph_conv()
    # test_dense_sage_conv()
    test_dense_cheb_conv()
    # test_sequential()
