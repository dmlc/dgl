import backend as F

import dgl
import dgl.function as fn
import dgl.nn.mxnet as nn
import mxnet as mx
import networkx as nx
import numpy as np
import pytest
import scipy as sp
from mxnet import autograd, gluon, nd
from utils import parametrize_idtype
from utils.graph_cases import (
    get_cases,
    random_bipartite,
    random_dglgraph,
    random_graph,
)


def check_close(a, b):
    assert np.allclose(a.asnumpy(), b.asnumpy(), rtol=1e-4, atol=1e-4)


def _AXWb(A, X, W, b):
    X = mx.nd.dot(X, W.data(X.context))
    Y = mx.nd.dot(A, X.reshape(X.shape[0], -1)).reshape(X.shape)
    return Y + b.data(X.context)


@parametrize_idtype
@pytest.mark.parametrize("out_dim", [1, 2])
def test_graph_conv(idtype, out_dim):
    g = dgl.from_networkx(nx.path_graph(3))
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    adj = g.adj_external(transpose=True, ctx=ctx)

    conv = nn.GraphConv(5, out_dim, norm="none", bias=True)
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

    conv = nn.GraphConv(5, out_dim)
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

    conv = nn.GraphConv(5, out_dim)
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
    check_close(g.ndata["h"], 2 * F.ones((3, 1)))


@parametrize_idtype
@pytest.mark.parametrize(
    "g",
    get_cases(["homo", "block-bipartite"], exclude=["zero-degree", "dglgraph"]),
)
@pytest.mark.parametrize("norm", ["none", "both", "right", "left"])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize("out_dim", [1, 2])
def test_graph_conv2(idtype, g, norm, weight, bias, out_dim):
    g = g.astype(idtype).to(F.ctx())
    conv = nn.GraphConv(5, out_dim, norm=norm, weight=weight, bias=bias)
    conv.initialize(ctx=F.ctx())
    ext_w = F.randn((5, out_dim)).as_in_context(F.ctx())
    nsrc = g.number_of_src_nodes()
    ndst = g.number_of_dst_nodes()
    h = F.randn((nsrc, 5)).as_in_context(F.ctx())
    if weight:
        h_out = conv(g, h)
    else:
        h_out = conv(g, h, ext_w)
    assert h_out.shape == (ndst, out_dim)


@parametrize_idtype
@pytest.mark.parametrize(
    "g", get_cases(["bipartite"], exclude=["zero-degree", "dglgraph"])
)
@pytest.mark.parametrize("norm", ["none", "both", "right"])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize("out_dim", [1, 2])
def test_graph_conv2_bi(idtype, g, norm, weight, bias, out_dim):
    g = g.astype(idtype).to(F.ctx())
    conv = nn.GraphConv(5, out_dim, norm=norm, weight=weight, bias=bias)
    conv.initialize(ctx=F.ctx())
    ext_w = F.randn((5, out_dim)).as_in_context(F.ctx())
    nsrc = g.number_of_src_nodes()
    ndst = g.number_of_dst_nodes()
    h = F.randn((nsrc, 5)).as_in_context(F.ctx())
    h_dst = F.randn((ndst, out_dim)).as_in_context(F.ctx())
    if weight:
        h_out = conv(g, (h, h_dst))
    else:
        h_out = conv(g, (h, h_dst), ext_w)
    assert h_out.shape == (ndst, out_dim)


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


@pytest.mark.parametrize("out_dim", [1, 2])
def test_tagconv(out_dim):
    g = dgl.from_networkx(nx.path_graph(3)).to(F.ctx())
    ctx = F.ctx()
    adj = g.adj_external(transpose=True, ctx=ctx)
    norm = mx.nd.power(g.in_degrees().astype("float32"), -0.5)

    conv = nn.TAGConv(5, out_dim, bias=True)
    conv.initialize(ctx=ctx)
    print(conv)

    # test#1: basic
    h0 = F.ones((3, 5))
    h1 = conv(g, h0)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    shp = norm.shape + (1,) * (h0.ndim - 1)
    norm = norm.reshape(shp).as_in_context(h0.context)

    assert F.allclose(
        h1, _S2AXWb(adj, norm, h0, conv.lin.data(ctx), conv.h_bias.data(ctx))
    )

    conv = nn.TAGConv(5, out_dim)
    conv.initialize(ctx=ctx)

    # test#2: basic
    h0 = F.ones((3, 5))
    h1 = conv(g, h0)
    assert h1.shape[-1] == out_dim


@parametrize_idtype
@pytest.mark.parametrize(
    "g", get_cases(["homo", "block-bipartite"], exclude=["zero-degree"])
)
@pytest.mark.parametrize("out_dim", [1, 20])
@pytest.mark.parametrize("num_heads", [1, 5])
def test_gat_conv(g, idtype, out_dim, num_heads):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    gat = nn.GATConv(10, out_dim, num_heads)  # n_heads = 5
    gat.initialize(ctx=ctx)
    print(gat)
    feat = F.randn((g.number_of_src_nodes(), 10))
    h = gat(g, feat)
    assert h.shape == (g.number_of_dst_nodes(), num_heads, out_dim)
    _, a = gat(g, feat, True)
    assert a.shape == (g.num_edges(), num_heads, 1)

    # test residual connection
    gat = nn.GATConv(10, out_dim, num_heads, residual=True)
    gat.initialize(ctx=ctx)
    h = gat(g, feat)


@parametrize_idtype
@pytest.mark.parametrize("g", get_cases(["bipartite"], exclude=["zero-degree"]))
@pytest.mark.parametrize("out_dim", [1, 2])
@pytest.mark.parametrize("num_heads", [1, 4])
def test_gat_conv_bi(g, idtype, out_dim, num_heads):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    gat = nn.GATConv(5, out_dim, num_heads)
    gat.initialize(ctx=ctx)
    feat = (
        F.randn((g.number_of_src_nodes(), 5)),
        F.randn((g.number_of_dst_nodes(), 5)),
    )
    h = gat(g, feat)
    assert h.shape == (g.number_of_dst_nodes(), num_heads, out_dim)
    _, a = gat(g, feat, True)
    assert a.shape == (g.num_edges(), num_heads, 1)


@parametrize_idtype
@pytest.mark.parametrize("g", get_cases(["homo", "block-bipartite"]))
@pytest.mark.parametrize("aggre_type", ["mean", "pool", "gcn"])
@pytest.mark.parametrize("out_dim", [1, 10])
def test_sage_conv(idtype, g, aggre_type, out_dim):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    sage = nn.SAGEConv(5, out_dim, aggre_type)
    feat = F.randn((g.number_of_src_nodes(), 5))
    sage.initialize(ctx=ctx)
    h = sage(g, feat)
    assert h.shape[-1] == out_dim


@parametrize_idtype
@pytest.mark.parametrize("g", get_cases(["bipartite"]))
@pytest.mark.parametrize("aggre_type", ["mean", "pool", "gcn"])
@pytest.mark.parametrize("out_dim", [1, 2])
def test_sage_conv_bi(idtype, g, aggre_type, out_dim):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    dst_dim = 5 if aggre_type != "gcn" else 10
    sage = nn.SAGEConv((10, dst_dim), out_dim, aggre_type)
    feat = (
        F.randn((g.number_of_src_nodes(), 10)),
        F.randn((g.number_of_dst_nodes(), dst_dim)),
    )
    sage.initialize(ctx=ctx)
    h = sage(g, feat)
    assert h.shape[-1] == out_dim
    assert h.shape[0] == g.number_of_dst_nodes()


@parametrize_idtype
@pytest.mark.parametrize("aggre_type", ["mean", "pool", "gcn"])
@pytest.mark.parametrize("out_dim", [1, 2])
def test_sage_conv_bi2(idtype, aggre_type, out_dim):
    # Test the case for graphs without edges
    g = dgl.heterograph({("_U", "_E", "_V"): ([], [])}, {"_U": 5, "_V": 3})
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    sage = nn.SAGEConv((3, 3), out_dim, "gcn")
    feat = (F.randn((5, 3)), F.randn((3, 3)))
    sage.initialize(ctx=ctx)
    h = sage(g, feat)
    assert h.shape[-1] == out_dim
    assert h.shape[0] == 3
    for aggre_type in ["mean", "pool"]:
        sage = nn.SAGEConv((3, 1), out_dim, aggre_type)
        feat = (F.randn((5, 3)), F.randn((3, 1)))
        sage.initialize(ctx=ctx)
        h = sage(g, feat)
        assert h.shape[-1] == out_dim
        assert h.shape[0] == 3


def test_gg_conv():
    g = dgl.from_networkx(nx.erdos_renyi_graph(20, 0.3)).to(F.ctx())
    ctx = F.ctx()

    gg_conv = nn.GatedGraphConv(10, 20, 3, 4)  # n_step = 3, n_etypes = 4
    gg_conv.initialize(ctx=ctx)
    print(gg_conv)

    # test#1: basic
    h0 = F.randn((20, 10))
    etypes = nd.random.randint(0, 4, g.num_edges()).as_in_context(ctx)
    h1 = gg_conv(g, h0, etypes)
    assert h1.shape == (20, 20)


@pytest.mark.parametrize("out_dim", [1, 20])
def test_cheb_conv(out_dim):
    g = dgl.from_networkx(nx.erdos_renyi_graph(20, 0.3)).to(F.ctx())
    ctx = F.ctx()

    cheb = nn.ChebConv(10, out_dim, 3)  # k = 3
    cheb.initialize(ctx=ctx)
    print(cheb)

    # test#1: basic
    h0 = F.randn((20, 10))
    h1 = cheb(g, h0)
    assert h1.shape == (20, out_dim)


@parametrize_idtype
@pytest.mark.parametrize(
    "g", get_cases(["homo", "block-bipartite"], exclude=["zero-degree"])
)
def test_agnn_conv(g, idtype):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    agnn_conv = nn.AGNNConv(0.1, True)
    agnn_conv.initialize(ctx=ctx)
    print(agnn_conv)
    feat = F.randn((g.number_of_src_nodes(), 10))
    h = agnn_conv(g, feat)
    assert h.shape == (g.number_of_dst_nodes(), 10)


@parametrize_idtype
@pytest.mark.parametrize("g", get_cases(["bipartite"], exclude=["zero-degree"]))
def test_agnn_conv_bi(g, idtype):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    agnn_conv = nn.AGNNConv(0.1, True)
    agnn_conv.initialize(ctx=ctx)
    print(agnn_conv)
    feat = (
        F.randn((g.number_of_src_nodes(), 5)),
        F.randn((g.number_of_dst_nodes(), 5)),
    )
    h = agnn_conv(g, feat)
    assert h.shape == (g.number_of_dst_nodes(), 5)


def test_appnp_conv():
    g = dgl.from_networkx(nx.erdos_renyi_graph(20, 0.3)).to(F.ctx())
    ctx = F.ctx()

    appnp_conv = nn.APPNPConv(3, 0.1, 0)
    appnp_conv.initialize(ctx=ctx)
    print(appnp_conv)

    # test#1: basic
    h0 = F.randn((20, 10))
    h1 = appnp_conv(g, h0)
    assert h1.shape == (20, 10)


@pytest.mark.parametrize("out_dim", [1, 2])
def test_dense_cheb_conv(out_dim):
    for k in range(1, 4):
        ctx = F.ctx()
        g = dgl.from_scipy(sp.sparse.random(100, 100, density=0.3)).to(F.ctx())
        adj = g.adj_external(transpose=True, ctx=ctx).tostype("default")
        cheb = nn.ChebConv(5, out_dim, k)
        dense_cheb = nn.DenseChebConv(5, out_dim, k)
        cheb.initialize(ctx=ctx)
        dense_cheb.initialize(ctx=ctx)

        for i in range(len(cheb.fc)):
            dense_cheb.fc[i].weight.set_data(cheb.fc[i].weight.data())
            if cheb.bias is not None:
                dense_cheb.bias.set_data(cheb.bias.data())

        feat = F.randn((100, 5))
        out_cheb = cheb(g, feat, [2.0])
        out_dense_cheb = dense_cheb(adj, feat, 2.0)
        assert F.allclose(out_cheb, out_dense_cheb)


@parametrize_idtype
@pytest.mark.parametrize("norm_type", ["both", "right", "none"])
@pytest.mark.parametrize(
    "g", get_cases(["homo", "block-bipartite"], exclude=["zero-degree"])
)
@pytest.mark.parametrize("out_dim", [1, 2])
def test_dense_graph_conv(idtype, g, norm_type, out_dim):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    adj = g.adj_external(transpose=True, ctx=ctx).tostype("default")
    conv = nn.GraphConv(5, out_dim, norm=norm_type, bias=True)
    dense_conv = nn.DenseGraphConv(5, out_dim, norm=norm_type, bias=True)
    conv.initialize(ctx=ctx)
    dense_conv.initialize(ctx=ctx)
    dense_conv.weight.set_data(conv.weight.data())
    dense_conv.bias.set_data(conv.bias.data())
    feat = F.randn((g.number_of_src_nodes(), 5))
    out_conv = conv(g, feat)
    out_dense_conv = dense_conv(adj, feat)
    assert F.allclose(out_conv, out_dense_conv)


@parametrize_idtype
@pytest.mark.parametrize(
    "g", get_cases(["homo", "bipartite", "block-bipartite"])
)
@pytest.mark.parametrize("out_dim", [1, 2])
def test_dense_sage_conv(idtype, g, out_dim):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    adj = g.adj_external(transpose=True, ctx=ctx).tostype("default")
    sage = nn.SAGEConv(5, out_dim, "gcn")
    dense_sage = nn.DenseSAGEConv(5, out_dim)
    sage.initialize(ctx=ctx)
    dense_sage.initialize(ctx=ctx)
    dense_sage.fc.weight.set_data(sage.fc_neigh.weight.data())
    dense_sage.fc.bias.set_data(sage.fc_neigh.bias.data())
    if len(g.ntypes) == 2:
        feat = (
            F.randn((g.number_of_src_nodes(), 5)),
            F.randn((g.number_of_dst_nodes(), 5)),
        )
    else:
        feat = F.randn((g.num_nodes(), 5))

    out_sage = sage(g, feat)
    out_dense_sage = dense_sage(adj, feat)
    assert F.allclose(out_sage, out_dense_sage)


@parametrize_idtype
@pytest.mark.parametrize(
    "g", get_cases(["homo", "block-bipartite"], exclude=["zero-degree"])
)
@pytest.mark.parametrize("out_dim", [1, 2])
def test_edge_conv(g, idtype, out_dim):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    edge_conv = nn.EdgeConv(5, out_dim)
    edge_conv.initialize(ctx=ctx)
    print(edge_conv)
    # test #1: basic
    h0 = F.randn((g.number_of_src_nodes(), 5))
    h1 = edge_conv(g, h0)
    assert h1.shape == (g.number_of_dst_nodes(), out_dim)


@parametrize_idtype
@pytest.mark.parametrize("g", get_cases(["bipartite"], exclude=["zero-degree"]))
@pytest.mark.parametrize("out_dim", [1, 2])
def test_edge_conv_bi(g, idtype, out_dim):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    edge_conv = nn.EdgeConv(5, out_dim)
    edge_conv.initialize(ctx=ctx)
    print(edge_conv)
    # test #1: basic
    h0 = F.randn((g.number_of_src_nodes(), 5))
    x0 = F.randn((g.number_of_dst_nodes(), 5))
    h1 = edge_conv(g, (h0, x0))
    assert h1.shape == (g.number_of_dst_nodes(), out_dim)


@parametrize_idtype
@pytest.mark.parametrize("g", get_cases(["homo", "block-bipartite"]))
@pytest.mark.parametrize("aggregator_type", ["mean", "max", "sum"])
def test_gin_conv(g, idtype, aggregator_type):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()

    gin_conv = nn.GINConv(lambda x: x, aggregator_type, 0.1)
    gin_conv.initialize(ctx=ctx)
    print(gin_conv)

    # test #1: basic
    feat = F.randn((g.number_of_src_nodes(), 5))
    h = gin_conv(g, feat)
    assert h.shape == (g.number_of_dst_nodes(), 5)


@parametrize_idtype
@pytest.mark.parametrize("g", get_cases(["bipartite"]))
@pytest.mark.parametrize("aggregator_type", ["mean", "max", "sum"])
def test_gin_conv_bi(g, idtype, aggregator_type):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()

    gin_conv = nn.GINConv(lambda x: x, aggregator_type, 0.1)
    gin_conv.initialize(ctx=ctx)
    print(gin_conv)

    # test #2: bipartite
    feat = (
        F.randn((g.number_of_src_nodes(), 5)),
        F.randn((g.number_of_dst_nodes(), 5)),
    )
    h = gin_conv(g, feat)
    return h.shape == (g.number_of_dst_nodes(), 5)


@parametrize_idtype
@pytest.mark.parametrize(
    "g", get_cases(["homo", "block-bipartite"], exclude=["zero-degree"])
)
def test_gmm_conv(g, idtype):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    gmm_conv = nn.GMMConv(5, 2, 5, 3, "max")
    gmm_conv.initialize(ctx=ctx)
    h0 = F.randn((g.number_of_src_nodes(), 5))
    pseudo = F.randn((g.num_edges(), 5))
    h1 = gmm_conv(g, h0, pseudo)
    assert h1.shape == (g.number_of_dst_nodes(), 2)


@parametrize_idtype
@pytest.mark.parametrize("g", get_cases(["bipartite"], exclude=["zero-degree"]))
def test_gmm_conv_bi(g, idtype):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    gmm_conv = nn.GMMConv((5, 4), 2, 5, 3, "max")
    gmm_conv.initialize(ctx=ctx)
    # test #1: basic
    h0 = F.randn((g.number_of_src_nodes(), 5))
    hd = F.randn((g.number_of_dst_nodes(), 4))
    pseudo = F.randn((g.num_edges(), 5))
    h1 = gmm_conv(g, (h0, hd), pseudo)
    assert h1.shape == (g.number_of_dst_nodes(), 2)


@parametrize_idtype
@pytest.mark.parametrize("g", get_cases(["homo", "block-bipartite"]))
def test_nn_conv(g, idtype):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    nn_conv = nn.NNConv(5, 2, gluon.nn.Embedding(3, 5 * 2), "max")
    nn_conv.initialize(ctx=ctx)
    # test #1: basic
    h0 = F.randn((g.number_of_src_nodes(), 5))
    etypes = nd.random.randint(0, 4, g.num_edges()).as_in_context(ctx)
    h1 = nn_conv(g, h0, etypes)
    assert h1.shape == (g.number_of_dst_nodes(), 2)


@parametrize_idtype
@pytest.mark.parametrize("g", get_cases(["bipartite"]))
def test_nn_conv_bi(g, idtype):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    nn_conv = nn.NNConv((5, 4), 2, gluon.nn.Embedding(3, 5 * 2), "max")
    nn_conv.initialize(ctx=ctx)
    # test #1: basic
    h0 = F.randn((g.number_of_src_nodes(), 5))
    hd = F.randn((g.number_of_dst_nodes(), 4))
    etypes = nd.random.randint(0, 4, g.num_edges()).as_in_context(ctx)
    h1 = nn_conv(g, (h0, hd), etypes)
    assert h1.shape == (g.number_of_dst_nodes(), 2)


@pytest.mark.parametrize("out_dim", [1, 2])
def test_sg_conv(out_dim):
    g = dgl.from_networkx(nx.erdos_renyi_graph(20, 0.3)).to(F.ctx())
    g = dgl.add_self_loop(g)
    ctx = F.ctx()

    sgc = nn.SGConv(5, out_dim, 2)
    sgc.initialize(ctx=ctx)
    print(sgc)

    # test #1: basic
    h0 = F.randn((g.num_nodes(), 5))
    h1 = sgc(g, h0)
    assert h1.shape == (g.num_nodes(), out_dim)


def test_set2set():
    g = dgl.from_networkx(nx.path_graph(10)).to(F.ctx())
    ctx = F.ctx()

    s2s = nn.Set2Set(5, 3, 3)  # hidden size 5, 3 iters, 3 layers
    s2s.initialize(ctx=ctx)
    print(s2s)

    # test#1: basic
    h0 = F.randn((g.num_nodes(), 5))
    h1 = s2s(g, h0)
    assert h1.shape[0] == 1 and h1.shape[1] == 10 and h1.ndim == 2

    # test#2: batched graph
    bg = dgl.batch([g, g, g])
    h0 = F.randn((bg.num_nodes(), 5))
    h1 = s2s(bg, h0)
    assert h1.shape[0] == 3 and h1.shape[1] == 10 and h1.ndim == 2


def test_glob_att_pool():
    g = dgl.from_networkx(nx.path_graph(10)).to(F.ctx())
    ctx = F.ctx()

    gap = nn.GlobalAttentionPooling(gluon.nn.Dense(1), gluon.nn.Dense(10))
    gap.initialize(ctx=ctx)
    print(gap)
    # test#1: basic
    h0 = F.randn((g.num_nodes(), 5))
    h1 = gap(g, h0)
    assert h1.shape[0] == 1 and h1.shape[1] == 10 and h1.ndim == 2

    # test#2: batched graph
    bg = dgl.batch([g, g, g, g])
    h0 = F.randn((bg.num_nodes(), 5))
    h1 = gap(bg, h0)
    assert h1.shape[0] == 4 and h1.shape[1] == 10 and h1.ndim == 2


def test_simple_pool():
    g = dgl.from_networkx(nx.path_graph(15)).to(F.ctx())

    sum_pool = nn.SumPooling()
    avg_pool = nn.AvgPooling()
    max_pool = nn.MaxPooling()
    sort_pool = nn.SortPooling(10)  # k = 10
    print(sum_pool, avg_pool, max_pool, sort_pool)

    # test#1: basic
    h0 = F.randn((g.num_nodes(), 5))
    h1 = sum_pool(g, h0)
    check_close(F.squeeze(h1, 0), F.sum(h0, 0))
    h1 = avg_pool(g, h0)
    check_close(F.squeeze(h1, 0), F.mean(h0, 0))
    h1 = max_pool(g, h0)
    check_close(F.squeeze(h1, 0), F.max(h0, 0))
    h1 = sort_pool(g, h0)
    assert h1.shape[0] == 1 and h1.shape[1] == 10 * 5 and h1.ndim == 2

    # test#2: batched graph
    g_ = dgl.from_networkx(nx.path_graph(5)).to(F.ctx())
    bg = dgl.batch([g, g_, g, g_, g])
    h0 = F.randn((bg.num_nodes(), 5))
    h1 = sum_pool(bg, h0)
    truth = mx.nd.stack(
        F.sum(h0[:15], 0),
        F.sum(h0[15:20], 0),
        F.sum(h0[20:35], 0),
        F.sum(h0[35:40], 0),
        F.sum(h0[40:55], 0),
        axis=0,
    )
    check_close(h1, truth)

    h1 = avg_pool(bg, h0)
    truth = mx.nd.stack(
        F.mean(h0[:15], 0),
        F.mean(h0[15:20], 0),
        F.mean(h0[20:35], 0),
        F.mean(h0[35:40], 0),
        F.mean(h0[40:55], 0),
        axis=0,
    )
    check_close(h1, truth)

    h1 = max_pool(bg, h0)
    truth = mx.nd.stack(
        F.max(h0[:15], 0),
        F.max(h0[15:20], 0),
        F.max(h0[20:35], 0),
        F.max(h0[35:40], 0),
        F.max(h0[40:55], 0),
        axis=0,
    )
    check_close(h1, truth)

    h1 = sort_pool(bg, h0)
    assert h1.shape[0] == 5 and h1.shape[1] == 10 * 5 and h1.ndim == 2


@pytest.mark.parametrize("O", [1, 2, 8])
def test_rgcn(O):
    ctx = F.ctx()
    etype = []
    g = dgl.from_scipy(sp.sparse.random(100, 100, density=0.1)).to(F.ctx())
    # 5 etypes
    R = 5
    for i in range(g.num_edges()):
        etype.append(i % 5)
    B = 2
    I = 10

    rgc_basis = nn.RelGraphConv(I, O, R, "basis", B)
    rgc_basis.initialize(ctx=ctx)
    h = nd.random.randn(100, I, ctx=ctx)
    r = nd.array(etype, ctx=ctx)
    h_new = rgc_basis(g, h, r)
    assert list(h_new.shape) == [100, O]

    if O % B == 0:
        rgc_bdd = nn.RelGraphConv(I, O, R, "bdd", B)
        rgc_bdd.initialize(ctx=ctx)
        h = nd.random.randn(100, I, ctx=ctx)
        r = nd.array(etype, ctx=ctx)
        h_new = rgc_bdd(g, h, r)
        assert list(h_new.shape) == [100, O]

    # with norm
    norm = nd.zeros((g.num_edges(), 1), ctx=ctx)

    rgc_basis = nn.RelGraphConv(I, O, R, "basis", B)
    rgc_basis.initialize(ctx=ctx)
    h = nd.random.randn(100, I, ctx=ctx)
    r = nd.array(etype, ctx=ctx)
    h_new = rgc_basis(g, h, r, norm)
    assert list(h_new.shape) == [100, O]

    if O % B == 0:
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
            graph.ndata["h"] = n_feat
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            n_feat += graph.ndata["h"]
            graph.apply_edges(fn.u_add_v("h", "h", "e"))
            e_feat += graph.edata["e"]
            return n_feat, e_feat

    g = dgl.graph(([], [])).to(F.ctx())
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
            graph.ndata["h"] = n_feat
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            n_feat += graph.ndata["h"]
            return n_feat.reshape(graph.num_nodes() // 2, 2, -1).sum(1)

    g1 = dgl.from_networkx(nx.erdos_renyi_graph(32, 0.05)).to(F.ctx())
    g2 = dgl.from_networkx(nx.erdos_renyi_graph(16, 0.2)).to(F.ctx())
    g3 = dgl.from_networkx(nx.erdos_renyi_graph(8, 0.8)).to(F.ctx())

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


@parametrize_idtype
@pytest.mark.parametrize("agg", ["sum", "max", "min", "mean", "stack", myagg])
def test_hetero_conv(agg, idtype):
    g = dgl.heterograph(
        {
            ("user", "follows", "user"): ([0, 0, 2, 1], [1, 2, 1, 3]),
            ("user", "plays", "game"): ([0, 0, 0, 1, 2], [0, 2, 3, 0, 2]),
            ("store", "sells", "game"): ([0, 0, 1, 1], [0, 3, 1, 2]),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    conv = nn.HeteroGraphConv(
        {
            "follows": nn.GraphConv(2, 3, allow_zero_in_degree=True),
            "plays": nn.GraphConv(2, 4, allow_zero_in_degree=True),
            "sells": nn.GraphConv(3, 4, allow_zero_in_degree=True),
        },
        agg,
    )
    conv.initialize(ctx=F.ctx())
    print(conv)
    uf = F.randn((4, 2))
    gf = F.randn((4, 4))
    sf = F.randn((2, 3))

    h = conv(g, {"user": uf, "store": sf, "game": gf})
    assert set(h.keys()) == {"user", "game"}
    if agg != "stack":
        assert h["user"].shape == (4, 3)
        assert h["game"].shape == (4, 4)
    else:
        assert h["user"].shape == (4, 1, 3)
        assert h["game"].shape == (4, 2, 4)

    block = dgl.to_block(
        g.to(F.cpu()), {"user": [0, 1, 2, 3], "game": [0, 1, 2, 3], "store": []}
    ).to(F.ctx())
    h = conv(
        block,
        (
            {"user": uf, "game": gf, "store": sf},
            {"user": uf, "game": gf, "store": sf[0:0]},
        ),
    )
    assert set(h.keys()) == {"user", "game"}
    if agg != "stack":
        assert h["user"].shape == (4, 3)
        assert h["game"].shape == (4, 4)
    else:
        assert h["user"].shape == (4, 1, 3)
        assert h["game"].shape == (4, 2, 4)

    h = conv(block, {"user": uf, "game": gf, "store": sf})
    assert set(h.keys()) == {"user", "game"}
    if agg != "stack":
        assert h["user"].shape == (4, 3)
        assert h["game"].shape == (4, 4)
    else:
        assert h["user"].shape == (4, 1, 3)
        assert h["game"].shape == (4, 2, 4)

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
    conv = nn.HeteroGraphConv(
        {"follows": mod1, "plays": mod2, "sells": mod3}, agg
    )
    conv.initialize(ctx=F.ctx())
    mod_args = {"follows": (1,), "plays": (1,)}
    h = conv(g, {"user": uf, "store": sf, "game": gf}, mod_args)
    assert mod1.carg1 == 1
    assert mod2.carg1 == 1
    assert mod3.carg1 == 0

    # conv on graph without any edges
    for etype in g.etypes:
        g = dgl.remove_edges(g, g.edges(form="eid", etype=etype), etype=etype)
    assert g.num_edges() == 0
    h = conv(g, {"user": uf, "game": gf, "store": sf})
    assert set(h.keys()) == {"user", "game"}

    block = dgl.to_block(
        g.to(F.cpu()), {"user": [0, 1, 2, 3], "game": [0, 1, 2, 3], "store": []}
    ).to(F.ctx())
    h = conv(
        block,
        (
            {"user": uf, "game": gf, "store": sf},
            {"user": uf, "game": gf, "store": sf[0:0]},
        ),
    )
    assert set(h.keys()) == {"user", "game"}


if __name__ == "__main__":
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
    test_set2set()
    test_glob_att_pool()
    test_simple_pool()
    test_rgcn()
    test_sequential()
    test_hetero_conv()
