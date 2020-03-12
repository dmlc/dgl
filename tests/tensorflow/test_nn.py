import tensorflow as tf
from tensorflow.keras import layers
import networkx as nx
import dgl
import dgl.nn.tensorflow as nn
import dgl.function as fn
import backend as F
from copy import deepcopy

import numpy as np
import scipy as sp

def _AXWb(A, X, W, b):
    X = tf.matmul(X, W)
    Y = tf.reshape(tf.matmul(A, tf.reshape(X, (X.shape[0], -1))), X.shape)
    return Y + b

def test_graph_conv():
    g = dgl.DGLGraph(nx.path_graph(3))
    ctx = F.ctx()
    adj = tf.sparse.to_dense(tf.sparse.reorder(g.adjacency_matrix(ctx=ctx)))

    conv = nn.GraphConv(5, 2, norm=False, bias=True)
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
    h1 = sum_pool(g, h0)
    assert F.allclose(F.squeeze(h1, 0), F.sum(h0, 0))
    h1 = avg_pool(g, h0)
    assert F.allclose(F.squeeze(h1, 0), F.mean(h0, 0))
    h1 = max_pool(g, h0)
    assert F.allclose(F.squeeze(h1, 0), F.max(h0, 0))
    h1 = sort_pool(g, h0)
    assert h1.shape[0] == 1 and h1.shape[1] == 10 * 5 and h1.ndim == 2

    # test#2: batched graph
    g_ = dgl.DGLGraph(nx.path_graph(5))
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

def uniform_attention(g, shape):
    a = F.ones(shape)
    target_shape = (g.number_of_edges(),) + (1,) * (len(shape) - 1)
    return a / tf.cast(tf.reshape(g.in_degrees(g.edges()[1]), target_shape), tf.float32)

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

    # Test both forward and backward with Tensorflow built-in softmax.
    g = dgl.DGLGraph()
    g.add_nodes(30)
    # build a complete graph
    for i in range(30):
        for j in range(30):
            g.add_edge(i, j)

    
    score = F.randn((900, 1))
    with tf.GradientTape() as tape:
        tape.watch(score)
        grad = F.randn((900, 1))
        y = tf.reshape(F.softmax(tf.reshape(score,(30, 30)), dim=0), (-1, 1))
        grads = tape.gradient(y, [score])
        grad_score = grads[0]

    with tf.GradientTape() as tape:
        tape.watch(score)
        y_dgl = nn.edge_softmax(g, score)
        assert len(g.ndata) == 0
        assert len(g.edata) == 0
        # check forward
        assert F.allclose(y_dgl, y)
        grads = tape.gradient(y_dgl, [score])
    # checkout gradient
    assert F.allclose(grads[0], grad_score)
    print(grads[0][:10], grad_score[:10])
    
    # Test 2
    def generate_rand_graph(n):
      arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
      return dgl.DGLGraph(arr, readonly=True)
    
    g = generate_rand_graph(50)
    a1 = F.randn((g.number_of_edges(), 1))
    a2 = tf.identity(a1)
    with tf.GradientTape() as tape:
        tape.watch(a1)
        g.edata['s'] = a1
        g.group_apply_edges('dst', lambda edges: {'ss':F.softmax(edges.data['s'], 1)})
        loss = tf.reduce_sum(g.edata['ss'])
        a1_grad = tape.gradient(loss, [a1])[0]
    
    with tf.GradientTape() as tape:
        tape.watch(a2)
        builtin_sm = nn.edge_softmax(g, a2)
        loss = tf.reduce_sum(builtin_sm)
        a2_grad = tape.gradient(loss, [a2])[0]
    print(a1_grad - a2_grad)
    assert len(g.ndata) == 0
    assert len(g.edata) == 2
    assert F.allclose(a1_grad, a2_grad, rtol=1e-4, atol=1e-4) # Follow tolerance in unittest backend

def test_partial_edge_softmax():
    g = dgl.DGLGraph()
    g.add_nodes(30)
    # build a complete graph
    for i in range(30):
        for j in range(30):
            g.add_edge(i, j)

    score = F.randn((300, 1))
    grad = F.randn((300, 1))
    import numpy as np
    eids = np.random.choice(900, 300, replace=False).astype('int64')
    eids = F.zerocopy_from_numpy(eids)
    # compute partial edge softmax
    with tf.GradientTape() as tape:
        tape.watch(score)
        y_1 = nn.edge_softmax(g, score, eids)
        grads = tape.gradient(y_1, [score])
    grad_1 = grads[0]
    # compute edge softmax on edge subgraph
    subg = g.edge_subgraph(eids)
    with tf.GradientTape() as tape:
        tape.watch(score)
        y_2 = nn.edge_softmax(subg, score)
        grads = tape.gradient(y_2, [score])
    grad_2 = grads[0]

    assert F.allclose(y_1, y_2)
    assert F.allclose(grad_1, grad_2)

def test_glob_att_pool():
    g = dgl.DGLGraph(nx.path_graph(10))

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
    g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
    # 5 etypes
    R = 5
    for i in range(g.number_of_edges()):
        etype.append(i % 5)
    B = 2
    I = 10
    O = 8

    rgc_basis = nn.RelGraphConv(I, O, R, "basis", B)
    h = tf.random.normal((100, I))
    r = tf.constant(etype)
    h_new = rgc_basis(g, h, r)
    assert list(h_new.shape) == [100, O]

    rgc_bdd = nn.RelGraphConv(I, O, R, "bdd", B)
    h = tf.random.normal((100, I))
    r = tf.constant(etype)
    h_new = rgc_bdd(g, h, r)
    assert list(h_new.shape) == [100, O]

    # with norm
    norm = tf.zeros((g.number_of_edges(), 1))

    rgc_basis = nn.RelGraphConv(I, O, R, "basis", B)
    h = tf.random.normal((100, I))
    r = tf.constant(etype)
    h_new = rgc_basis(g, h, r, norm)
    assert list(h_new.shape) == [100, O]

    rgc_bdd = nn.RelGraphConv(I, O, R, "bdd", B)
    h = tf.random.normal((100, I))
    r = tf.constant(etype)
    h_new = rgc_bdd(g, h, r, norm)
    assert list(h_new.shape) == [100, O]

    # id input
    rgc_basis = nn.RelGraphConv(I, O, R, "basis", B)
    h = tf.constant(np.random.randint(0, I, (100,)))
    r = tf.constant(etype)
    h_new = rgc_basis(g, h, r)
    assert list(h_new.shape) == [100, O]

def test_gat_conv():
    g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
    gat = nn.GATConv(5, 2, 4)
    feat = F.randn((100, 5))
    h = gat(g, feat)
    assert h.shape[-1] == 2 and h.shape[-2] == 4

def test_sage_conv():
    for aggre_type in ['mean', 'pool', 'gcn', 'lstm']:
        g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
        sage = nn.SAGEConv(5, 10, aggre_type)
        feat = F.randn((100, 5))
        h = sage(g, feat)
        assert h.shape[-1] == 10

def test_sgc_conv():
    ctx = F.ctx()
    g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
    # not cached
    sgc = nn.SGConv(5, 10, 3)
    feat = F.randn((100, 5))

    h = sgc(g, feat)
    assert h.shape[-1] == 10

    # cached
    sgc = nn.SGConv(5, 10, 3, True)
    h_0 = sgc(g, feat)
    h_1 = sgc(g, feat + 1)
    assert F.allclose(h_0, h_1)
    assert h_0.shape[-1] == 10

def test_appnp_conv():
    g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
    appnp = nn.APPNPConv(10, 0.1)
    feat = F.randn((100, 5))

    h = appnp(g, feat)
    assert h.shape[-1] == 5

def test_gin_conv():
    for aggregator_type in ['mean', 'max', 'sum']:
        g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
        gin = nn.GINConv(
            tf.keras.layers.Dense(12),
            aggregator_type
        )
        feat = F.randn((100, 5))
        gin = gin
        h = gin(g, feat)
        assert h.shape[-1] == 12


if __name__ == '__main__':
    test_graph_conv()
    test_edge_softmax()
    test_partial_edge_softmax()
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
    # test_dense_cheb_conv()
    # test_sequential()

