import backend as F
import numpy as np
import scipy as sp
import dgl
from dgl.contrib.sampling.sampler import create_full_nodeflow, NeighborSampler
from dgl import utils
import dgl.function as fn
from functools import partial
import itertools


def generate_rand_graph(n, connect_more=False, complete=False):
    if complete:
        cord = [(i,j) for i, j in itertools.product(range(n), range(n)) if i != j]
        row = [t[0] for t in cord]
        col = [t[1] for t in cord]
        data = np.ones((len(row),))
        arr = sp.sparse.coo_matrix((data, (row, col)), shape=(n, n))
    else:
        arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
        # having one node to connect to all other nodes.
        if connect_more:
            arr[0] = 1
            arr[:,0] = 1
    g = dgl.DGLGraph(arr, readonly=True)
    g.ndata['h1'] = F.randn((g.number_of_nodes(), 10))
    g.edata['h2'] = F.randn((g.number_of_edges(), 3))
    return g


def test_self_loop():
    n = 100
    num_hops = 2
    g = generate_rand_graph(n, complete=True)
    nf = create_mini_batch(g, num_hops, add_self_loop=True)
    for i in range(1, nf.num_layers):
        in_deg = nf.layer_in_degree(i)
        deg = F.ones(in_deg.shape, dtype=F.int64) * n
        assert F.array_equal(in_deg, deg)

def create_mini_batch(g, num_hops, add_self_loop=False):
    seed_ids = np.array([0, 1, 2, 3])
    sampler = NeighborSampler(g, batch_size=4, expand_factor=g.number_of_nodes(),
            num_hops=num_hops, seed_nodes=seed_ids, add_self_loop=add_self_loop)
    nfs = list(sampler)
    assert len(nfs) == 1
    return nfs[0]

def check_basic(g, nf):
    num_nodes = 0
    for i in range(nf.num_layers):
        num_nodes += nf.layer_size(i)
    assert nf.number_of_nodes() == num_nodes
    num_edges = 0
    for i in range(nf.num_blocks):
        num_edges += nf.block_size(i)
    assert nf.number_of_edges() == num_edges

    deg = nf.layer_in_degree(0)
    assert F.array_equal(deg, F.zeros((nf.layer_size(0)), F.int64))
    deg = nf.layer_out_degree(-1)
    assert F.array_equal(deg, F.zeros((nf.layer_size(-1)), F.int64))
    for i in range(1, nf.num_layers):
        in_deg = nf.layer_in_degree(i)
        out_deg = nf.layer_out_degree(i - 1)
        assert F.asnumpy(F.sum(in_deg, 0) == F.sum(out_deg, 0))


def test_basic():
    num_layers = 2
    g = generate_rand_graph(100, connect_more=True)
    nf = create_full_nodeflow(g, num_layers)
    assert nf.number_of_nodes() == g.number_of_nodes() * (num_layers + 1)
    assert nf.number_of_edges() == g.number_of_edges() * num_layers
    assert nf.num_layers == num_layers + 1
    assert nf.layer_size(0) == g.number_of_nodes()
    assert nf.layer_size(1) == g.number_of_nodes()
    check_basic(g, nf)

    parent_nids = F.arange(0, g.number_of_nodes())
    nids = nf.map_from_parent_nid(0, parent_nids)
    assert F.array_equal(nids, parent_nids)

    g = generate_rand_graph(100)
    nf = create_mini_batch(g, num_layers)
    assert nf.num_layers == num_layers + 1
    check_basic(g, nf)


def check_apply_nodes(create_node_flow):
    num_layers = 2
    for i in range(num_layers):
        g = generate_rand_graph(100)
        nf = create_node_flow(g, num_layers)
        nf.copy_from_parent()
        new_feats = F.randn((nf.layer_size(i), 5))
        def update_func(nodes):
            return {'h1' : new_feats}
        nf.apply_layer(i, update_func)
        assert F.array_equal(nf.layers[i].data['h1'], new_feats)

        new_feats = F.randn((4, 5))
        def update_func1(nodes):
            return {'h1' : new_feats}
        nf.apply_layer(i, update_func1, v=nf.layer_nid(i)[0:4])
        assert F.array_equal(nf.layers[i].data['h1'][0:4], new_feats)


def test_apply_nodes():
    check_apply_nodes(create_full_nodeflow)
    check_apply_nodes(create_mini_batch)


def check_apply_edges(create_node_flow):
    num_layers = 2
    for i in range(num_layers):
        g = generate_rand_graph(100)
        nf = create_node_flow(g, num_layers)
        nf.copy_from_parent()
        new_feats = F.randn((nf.block_size(i), 5))
        def update_func(nodes):
            return {'h2' : new_feats}
        nf.apply_block(i, update_func)
        assert F.array_equal(nf.blocks[i].data['h2'], new_feats)


def test_apply_edges():
    check_apply_edges(create_full_nodeflow)
    check_apply_edges(create_mini_batch)


def check_flow_compute(create_node_flow):
    num_layers = 2
    g = generate_rand_graph(100)
    nf = create_node_flow(g, num_layers)
    nf.copy_from_parent()
    g.ndata['h'] = g.ndata['h1']
    nf.layers[0].data['h'] = nf.layers[0].data['h1']
    # Test the computation on a layer at a time.
    for i in range(num_layers):
        nf.block_compute(i, fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                         lambda nodes: {'h' : nodes.data['t'] + 1})
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                     lambda nodes: {'h' : nodes.data['t'] + 1})
        assert F.array_equal(nf.layers[i + 1].data['h'], g.ndata['h'][nf.layer_parent_nid(i + 1)])

    # Test the computation when only a few nodes are active in a layer.
    g.ndata['h'] = g.ndata['h1']
    for i in range(num_layers):
        vs = nf.layer_nid(i+1)[0:4]
        nf.block_compute(i, fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                        lambda nodes: {'h' : nodes.data['t'] + 1}, v=vs)
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                     lambda nodes: {'h' : nodes.data['t'] + 1})
        data1 = nf.layers[i + 1].data['h'][0:4]
        data2 = g.ndata['h'][nf.map_to_parent_nid(vs)]
        assert F.array_equal(data1, data2)


def test_flow_compute():
    check_flow_compute(create_full_nodeflow)
    check_flow_compute(create_mini_batch)


def check_prop_flows(create_node_flow):
    num_layers = 2
    g = generate_rand_graph(100)
    g.ndata['h'] = g.ndata['h1']
    nf2 = create_node_flow(g, num_layers)
    nf2.copy_from_parent()
    # Test the computation on a layer at a time.
    for i in range(num_layers):
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                     lambda nodes: {'h' : nodes.data['t'] + 1})

    # Test the computation on all layers.
    nf2.prop_flow(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                  lambda nodes: {'h' : nodes.data['t'] + 1})
    assert F.array_equal(nf2.layers[-1].data['h'], g.ndata['h'][nf2.layer_parent_nid(-1)])


def test_prop_flows():
    check_prop_flows(create_full_nodeflow)
    check_prop_flows(create_mini_batch)


def test_copy():
    num_layers = 2
    g = generate_rand_graph(100)
    g.ndata['h'] = g.ndata['h1']
    nf = create_mini_batch(g, num_layers)
    nf.copy_from_parent()
    for i in range(nf.num_layers):
        assert len(g.ndata.keys()) == len(nf.layers[i].data.keys())
        for key in g.ndata.keys():
            assert key in nf.layers[i].data.keys()
            assert F.array_equal(nf.layers[i].data[key], g.ndata[key][nf.layer_parent_nid(i)])
    for i in range(nf.num_blocks):
        assert len(g.edata.keys()) == len(nf.blocks[i].data.keys())
        for key in g.edata.keys():
            assert key in nf.blocks[i].data.keys()
            assert F.array_equal(nf.blocks[i].data[key], g.edata[key][nf.block_parent_eid(i)])

    nf = create_mini_batch(g, num_layers)
    node_embed_names = [['h'], ['h1'], ['h']]
    edge_embed_names = [['h2'], ['h2']]
    nf.copy_from_parent(node_embed_names=node_embed_names, edge_embed_names=edge_embed_names)
    for i in range(nf.num_layers):
        assert len(node_embed_names[i]) == len(nf.layers[i].data.keys())
        for key in node_embed_names[i]:
            assert key in nf.layers[i].data.keys()
            assert F.array_equal(nf.layers[i].data[key], g.ndata[key][nf.layer_parent_nid(i)])
    for i in range(nf.num_blocks):
        assert len(edge_embed_names[i]) == len(nf.blocks[i].data.keys())
        for key in edge_embed_names[i]:
            assert key in nf.blocks[i].data.keys()
            assert F.array_equal(nf.blocks[i].data[key], g.edata[key][nf.block_parent_eid(i)])

    nf = create_mini_batch(g, num_layers)
    g.ndata['h0'] = F.clone(g.ndata['h'])
    node_embed_names = [['h0'], [], []]
    nf.copy_from_parent(node_embed_names=node_embed_names, edge_embed_names=None)
    for i in range(num_layers):
        nf.block_compute(i, fn.copy_src(src='h%d' % i, out='m'), fn.sum(msg='m', out='t'),
                         lambda nodes: {'h%d' % (i+1) : nodes.data['t'] + 1})
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                     lambda nodes: {'h' : nodes.data['t'] + 1})
        assert F.array_equal(nf.layers[i + 1].data['h%d' % (i+1)],
                             g.ndata['h'][nf.layer_parent_nid(i + 1)])
    nf.copy_to_parent(node_embed_names=[['h0'], ['h1'], ['h2']])
    for i in range(num_layers + 1):
        assert F.array_equal(nf.layers[i].data['h%d' % i],
                             g.ndata['h%d' % i][nf.layer_parent_nid(i)])

    nf = create_mini_batch(g, num_layers)
    g.ndata['h0'] = F.clone(g.ndata['h'])
    g.ndata['h1'] = F.clone(g.ndata['h'])
    g.ndata['h2'] = F.clone(g.ndata['h'])
    node_embed_names = [['h0'], ['h1'], ['h2']]
    nf.copy_from_parent(node_embed_names=node_embed_names, edge_embed_names=None)

    def msg_func(edge, ind):
        assert 'h%d' % ind in edge.src.keys()
        return {'m' : edge.src['h%d' % ind]}
    def reduce_func(node, ind):
        assert 'h%d' % (ind + 1) in node.data.keys()
        return {'h' : F.sum(node.mailbox['m'], 1) + node.data['h%d' % (ind + 1)]}

    for i in range(num_layers):
        nf.block_compute(i, partial(msg_func, ind=i), partial(reduce_func, ind=i))


def test_block_adj_matrix():
    num_layers = 3
    g = generate_rand_graph(100)
    nf = create_mini_batch(g, num_layers)
    assert nf.num_layers == num_layers + 1
    for i in range(nf.num_blocks):
        src, dst, eid = nf.block_edges(i)
        dest_nodes = utils.toindex(nf.layer_nid(i + 1))
        u, v, _ = nf._graph.in_edges(dest_nodes)
        u = nf._glb2lcl_nid(u.tousertensor(), i)
        v = nf._glb2lcl_nid(v.tousertensor(), i + 1)
        assert F.array_equal(src, u)
        assert F.array_equal(dst, v)

        adj, _ = nf.block_adjacency_matrix(i, F.cpu())
        adj = F.sparse_to_numpy(adj)
        data = np.ones((len(u)), dtype=np.float32)
        v = utils.toindex(v)
        u = utils.toindex(u)
        coo = sp.sparse.coo_matrix((data, (v.tonumpy(), u.tonumpy())),
                                   shape=adj.shape).todense()
        assert np.array_equal(adj, coo)


if __name__ == '__main__':
    test_basic()
    test_block_adj_matrix()
    test_copy()
    test_apply_nodes()
    test_apply_edges()
    test_flow_compute()
    test_prop_flows()
    test_self_loop()
