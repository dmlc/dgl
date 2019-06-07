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
        deg = F.copy_to(F.ones(in_deg.shape, dtype=F.int64), F.cpu()) * n
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
    assert F.array_equal(deg, F.copy_to(F.zeros((nf.layer_size(0)), F.int64), F.cpu()))
    deg = nf.layer_out_degree(-1)
    assert F.array_equal(deg, F.copy_to(F.zeros((nf.layer_size(-1)), F.int64), F.cpu()))
    for i in range(1, nf.num_layers):
        in_deg = nf.layer_in_degree(i)
        out_deg = nf.layer_out_degree(i - 1)
        assert F.asnumpy(F.sum(in_deg, 0) == F.sum(out_deg, 0))

    # negative layer Ids.
    for i in range(-1, -nf.num_layers, -1):
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

    parent_nids = F.copy_to(F.arange(0, g.number_of_nodes()), F.cpu())
    nids = nf.map_from_parent_nid(0, parent_nids)
    assert F.array_equal(nids, parent_nids)

    # should also work for negative layer ids
    for l in range(-1, -num_layers, -1):
        nids1 = nf.map_from_parent_nid(l, parent_nids)
        nids2 = nf.map_from_parent_nid(l + num_layers, parent_nids)
        assert F.array_equal(nids1, nids2)

    g = generate_rand_graph(100)
    nf = create_mini_batch(g, num_layers)
    assert nf.num_layers == num_layers + 1
    check_basic(g, nf)


def check_apply_nodes(create_node_flow, use_negative_block_id):
    num_layers = 2
    for i in range(num_layers):
        l = -num_layers + i if use_negative_block_id else i
        g = generate_rand_graph(100)
        nf = create_node_flow(g, num_layers)
        nf.copy_from_parent()
        new_feats = F.randn((nf.layer_size(l), 5))
        def update_func(nodes):
            return {'h1' : new_feats}
        nf.apply_layer(l, update_func)
        assert F.array_equal(nf.layers[l].data['h1'], new_feats)

        new_feats = F.randn((4, 5))
        def update_func1(nodes):
            return {'h1' : new_feats}
        nf.apply_layer(l, update_func1, v=nf.layer_nid(l)[0:4])
        assert F.array_equal(nf.layers[l].data['h1'][0:4], new_feats)


def test_apply_nodes():
    check_apply_nodes(create_full_nodeflow, use_negative_block_id=False)
    check_apply_nodes(create_mini_batch, use_negative_block_id=False)
    check_apply_nodes(create_full_nodeflow, use_negative_block_id=True)
    check_apply_nodes(create_mini_batch, use_negative_block_id=True)


def check_apply_edges(create_node_flow):
    num_layers = 2
    for i in range(num_layers):
        g = generate_rand_graph(100)
        g.ndata["f"] = F.randn((100, 10))
        nf = create_node_flow(g, num_layers)
        nf.copy_from_parent()
        new_feats = F.randn((nf.block_size(i), 5))

        def update_func(edges):
            return {'h2': new_feats, "f2": edges.src["f"] + edges.dst["f"]}

        nf.apply_block(i, update_func)
        assert F.array_equal(nf.blocks[i].data['h2'], new_feats)

        # should also work for negative block ids
        nf.apply_block(-num_layers + i, update_func)
        assert F.array_equal(nf.blocks[i].data['h2'], new_feats)

        eids = nf.block_parent_eid(i)
        srcs, dsts = g.find_edges(eids)
        expected_f_sum = g.nodes[srcs].data["f"] + g.nodes[dsts].data["f"]
        assert F.array_equal(nf.blocks[i].data['f2'], expected_f_sum)


def check_apply_edges1(create_node_flow):
    num_layers = 2
    for i in range(num_layers):
        g = generate_rand_graph(100)
        g.ndata["f"] = F.randn((100, 10))
        nf = create_node_flow(g, num_layers)
        nf.copy_from_parent()
        new_feats = F.randn((nf.block_size(i), 5))

        def update_func(edges):
            return {'h2': new_feats, "f2": edges.src["f"] + edges.dst["f"]}

        nf.register_apply_edge_func(update_func, i)
        nf.apply_block(i)
        assert F.array_equal(nf.blocks[i].data['h2'], new_feats)

        # should also work for negative block ids
        nf.register_apply_edge_func(update_func, -num_layers + i)
        nf.apply_block(-num_layers + i)
        assert F.array_equal(nf.blocks[i].data['h2'], new_feats)

        eids = nf.block_parent_eid(i)
        srcs, dsts = g.find_edges(eids)
        expected_f_sum = g.ndata["f"][srcs] + g.ndata["f"][dsts]
        assert F.array_equal(nf.blocks[i].data['f2'], expected_f_sum)


def test_apply_edges():
    check_apply_edges(create_full_nodeflow)
    check_apply_edges(create_mini_batch)
    check_apply_edges1(create_mini_batch)


def check_flow_compute(create_node_flow, use_negative_block_id=False):
    num_layers = 2
    g = generate_rand_graph(100)
    nf = create_node_flow(g, num_layers)
    nf.copy_from_parent()
    g.ndata['h'] = g.ndata['h1']
    nf.layers[0].data['h'] = nf.layers[0].data['h1']
    # Test the computation on a layer at a time.
    for i in range(num_layers):
        l = -num_layers + i if use_negative_block_id else i
        nf.block_compute(l, fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                         lambda nodes: {'h' : nodes.data['t'] + 1})
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                     lambda nodes: {'h' : nodes.data['t'] + 1})
        assert F.allclose(nf.layers[i + 1].data['h'], g.nodes[nf.layer_parent_nid(i + 1)].data['h'])

    # Test the computation when only a few nodes are active in a layer.
    g.ndata['h'] = g.ndata['h1']
    for i in range(num_layers):
        l = -num_layers + i if use_negative_block_id else i
        vs = nf.layer_nid(i+1)[0:4]
        nf.block_compute(l, fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                        lambda nodes: {'h' : nodes.data['t'] + 1}, v=vs)
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                     lambda nodes: {'h' : nodes.data['t'] + 1})
        data1 = nf.layers[i + 1].data['h'][0:4]
        data2 = g.nodes[nf.map_to_parent_nid(vs)].data['h']
        assert F.allclose(data1, data2)


def check_flow_compute1(create_node_flow, use_negative_block_id=False):
    num_layers = 2
    g = generate_rand_graph(100)

    # test the case that we register UDFs per block.
    nf = create_node_flow(g, num_layers)
    nf.copy_from_parent()
    g.ndata['h'] = g.ndata['h1']
    nf.layers[0].data['h'] = nf.layers[0].data['h1']
    for i in range(num_layers):
        l = -num_layers + i if use_negative_block_id else i
        nf.register_message_func(fn.copy_src(src='h', out='m'), l)
        nf.register_reduce_func(fn.sum(msg='m', out='t'), l)
        nf.register_apply_node_func(lambda nodes: {'h' : nodes.data['t'] + 1}, l)
        nf.block_compute(l)
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                     lambda nodes: {'h' : nodes.data['t'] + 1})
        assert F.allclose(nf.layers[i + 1].data['h'], g.ndata['h'][nf.layer_parent_nid(i + 1)])

    # test the case that we register UDFs in all blocks.
    nf = create_node_flow(g, num_layers)
    nf.copy_from_parent()
    g.ndata['h'] = g.ndata['h1']
    nf.layers[0].data['h'] = nf.layers[0].data['h1']
    nf.register_message_func(fn.copy_src(src='h', out='m'))
    nf.register_reduce_func(fn.sum(msg='m', out='t'))
    nf.register_apply_node_func(lambda nodes: {'h' : nodes.data['t'] + 1})
    for i in range(num_layers):
        l = -num_layers + i if use_negative_block_id else i
        nf.block_compute(l)
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                     lambda nodes: {'h' : nodes.data['t'] + 1})
        assert F.allclose(nf.layers[i + 1].data['h'], g.ndata['h'][nf.layer_parent_nid(i + 1)])


def test_flow_compute():
    check_flow_compute(create_full_nodeflow)
    check_flow_compute(create_mini_batch)
    check_flow_compute(create_full_nodeflow, use_negative_block_id=True)
    check_flow_compute(create_mini_batch, use_negative_block_id=True)
    check_flow_compute1(create_mini_batch)
    check_flow_compute1(create_mini_batch, use_negative_block_id=True)


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
    assert F.allclose(nf2.layers[-1].data['h'], g.nodes[nf2.layer_parent_nid(-1)].data['h'])


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
            assert F.array_equal(nf.layers[i].data[key], g.nodes[nf.layer_parent_nid(i)].data[key])
    for i in range(nf.num_blocks):
        assert len(g.edata.keys()) == len(nf.blocks[i].data.keys())
        for key in g.edata.keys():
            assert key in nf.blocks[i].data.keys()
            assert F.array_equal(nf.blocks[i].data[key], g.edges[nf.block_parent_eid(i)].data[key])

    nf = create_mini_batch(g, num_layers)
    node_embed_names = [['h'], ['h1'], ['h']]
    edge_embed_names = [['h2'], ['h2']]
    nf.copy_from_parent(node_embed_names=node_embed_names, edge_embed_names=edge_embed_names)
    for i in range(nf.num_layers):
        assert len(node_embed_names[i]) == len(nf.layers[i].data.keys())
        for key in node_embed_names[i]:
            assert key in nf.layers[i].data.keys()
            assert F.array_equal(nf.layers[i].data[key], g.nodes[nf.layer_parent_nid(i)].data[key])
    for i in range(nf.num_blocks):
        assert len(edge_embed_names[i]) == len(nf.blocks[i].data.keys())
        for key in edge_embed_names[i]:
            assert key in nf.blocks[i].data.keys()
            assert F.array_equal(nf.blocks[i].data[key], g.edges[nf.block_parent_eid(i)].data[key])

    nf = create_mini_batch(g, num_layers)
    g.ndata['h0'] = F.clone(g.ndata['h'])
    node_embed_names = [['h0'], [], []]
    nf.copy_from_parent(node_embed_names=node_embed_names, edge_embed_names=None)
    for i in range(num_layers):
        nf.block_compute(i, fn.copy_src(src='h%d' % i, out='m'), fn.sum(msg='m', out='t'),
                         lambda nodes: {'h%d' % (i+1) : nodes.data['t'] + 1})
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                     lambda nodes: {'h' : nodes.data['t'] + 1})
        assert F.allclose(nf.layers[i + 1].data['h%d' % (i+1)],
                          g.nodes[nf.layer_parent_nid(i + 1)].data['h'])
    nf.copy_to_parent(node_embed_names=[['h0'], ['h1'], ['h2']])
    for i in range(num_layers + 1):
        assert F.array_equal(nf.layers[i].data['h%d' % i],
                             g.nodes[nf.layer_parent_nid(i)].data['h%d' % i])

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


def test_block_edges():
    num_layers = 3
    g = generate_rand_graph(100)
    nf = create_mini_batch(g, num_layers)
    assert nf.num_layers == num_layers + 1
    for i in range(nf.num_blocks):
        src, dst, eid = nf.block_edges(i, remap=True)

        # should also work for negative block ids
        src_by_neg, dst_by_neg, eid_by_neg = nf.block_edges(-nf.num_blocks + i, remap=True)
        assert F.array_equal(src, src_by_neg)
        assert F.array_equal(dst, dst_by_neg)
        assert F.array_equal(eid, eid_by_neg)

        dest_nodes = utils.toindex(nf.layer_nid(i + 1))
        u, v, _ = nf._graph.in_edges(dest_nodes)
        u = nf._glb2lcl_nid(u.tousertensor(), i)
        v = nf._glb2lcl_nid(v.tousertensor(), i + 1)
        assert F.array_equal(src, u)
        assert F.array_equal(dst, v)


def test_block_adj_matrix():
    num_layers = 3
    g = generate_rand_graph(100)
    nf = create_mini_batch(g, num_layers)
    assert nf.num_layers == num_layers + 1
    for i in range(nf.num_blocks):
        u, v, _ = nf.block_edges(i, remap=True)
        adj, _ = nf.block_adjacency_matrix(i, F.cpu())
        adj = F.sparse_to_numpy(adj)

        # should also work for negative block ids
        adj_by_neg, _ = nf.block_adjacency_matrix(-nf.num_blocks + i, F.cpu())
        adj_by_neg = F.sparse_to_numpy(adj_by_neg)

        data = np.ones((len(u)), dtype=np.float32)
        v = utils.toindex(v)
        u = utils.toindex(u)
        coo = sp.sparse.coo_matrix((data, (v.tonumpy(), u.tonumpy())),
                                   shape=adj.shape).todense()
        assert np.array_equal(adj, coo)
        assert np.array_equal(adj_by_neg, coo)


def test_block_incidence_matrix():
    num_layers = 3
    g = generate_rand_graph(100)
    nf = create_mini_batch(g, num_layers)
    assert nf.num_layers == num_layers + 1
    for i in range(nf.num_blocks):
        typestrs = ["in", "out"] # todo need fix for "both"
        adjs = []
        for typestr in typestrs:
            adj, _ = nf.block_incidence_matrix(i, typestr, F.cpu())
            adj = F.sparse_to_numpy(adj)
            adjs.append(adj)

        # should work for negative block ids
        adjs_by_neg = []
        for typestr in typestrs:
            adj_by_neg, _ = nf.block_incidence_matrix(-nf.num_blocks + i, typestr, F.cpu())
            adj_by_neg = F.sparse_to_numpy(adj_by_neg)
            adjs_by_neg.append(adj_by_neg)

        u, v, e = nf.block_edges(i, remap=True)
        u = utils.toindex(u)
        v = utils.toindex(v)
        e = utils.toindex(e)

        expected = []
        data_in_and_out = np.ones((len(u)), dtype=np.float32)
        expected.append(
            sp.sparse.coo_matrix((data_in_and_out, (v.tonumpy(), e.tonumpy())),
                                 shape=adjs[0].shape).todense()
        )
        expected.append(
            sp.sparse.coo_matrix((data_in_and_out, (u.tonumpy(), e.tonumpy())),
                                 shape=adjs[1].shape).todense()
        )
        for i in range(len(typestrs)):
            assert np.array_equal(adjs[i], expected[i])
            assert np.array_equal(adjs_by_neg[i], expected[i])


if __name__ == '__main__':
    test_basic()
    test_block_adj_matrix()
    test_copy()
    test_apply_nodes()
    test_apply_edges()
    test_flow_compute()
    test_prop_flows()
    test_self_loop()
    test_block_edges()
    test_block_incidence_matrix()
