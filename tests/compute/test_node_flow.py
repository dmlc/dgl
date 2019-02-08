import backend as F
import numpy as np
import scipy as sp
import dgl
from dgl.node_flow import create_full_node_flow
from dgl import utils
import dgl.function as fn
from functools import partial

def generate_rand_graph(n, connect_more=False):
    arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
    # having one node to connect to all other nodes.
    if connect_more:
        arr[0] = 1
        arr[:,0] = 1
    g = dgl.DGLGraph(arr, readonly=True)
    g.ndata['h1'] = F.randn((g.number_of_nodes(), 10))
    g.edata['h2'] = F.randn((g.number_of_edges(), 3))
    return g


def create_mini_batch(g, num_hops):
    seed_ids = np.array([0, 1, 2, 3])
    seed_ids = utils.toindex(seed_ids)
    sgi = g._graph.neighbor_sampling([seed_ids], g.number_of_nodes(), num_hops, "in", None)
    assert len(sgi) == 1
    return dgl.node_flow.NodeFlow(g, sgi[0])


def check_basic(g, nf):
    num_nodes = 0
    for i in range(nf.num_layers):
        num_nodes += nf.layer_size(i)
    assert nf.number_of_nodes() == num_nodes
    num_edges = 0
    for i in range(nf.num_flows):
        num_edges += nf.flow_size(i)
    assert nf.number_of_edges() == num_edges

    deg = nf.layer_in_degree(0)
    assert F.array_equal(deg, F.zeros((nf.layer_size(0)), 'int64'))
    deg = nf.layer_out_degree(-1)
    assert F.array_equal(deg, F.zeros((nf.layer_size(-1)), 'int64'))
    for i in range(1, nf.num_layers):
        in_deg = nf.layer_in_degree(i)
        out_deg = nf.layer_out_degree(i - 1)
        assert F.asnumpy(F.sum(in_deg, 0) == F.sum(out_deg, 0))

    nf.copy_from_parent()
    assert F.array_equal(nf.layers[0].data['h1'], g.ndata['h1'][nf.layer_parent_nid(0)])
    assert F.array_equal(nf.layers[1].data['h1'], g.ndata['h1'][nf.layer_parent_nid(1)])
    assert F.array_equal(nf.layers[2].data['h1'], g.ndata['h1'][nf.layer_parent_nid(2)])
    assert F.array_equal(nf.flows[0].data['h2'], g.edata['h2'][nf.flow_parent_eid(0)])
    assert F.array_equal(nf.flows[1].data['h2'], g.edata['h2'][nf.flow_parent_eid(1)])

    for i in range(nf.number_of_nodes()):
        layer_id, local_nid = nf.map_to_layer_nid(i)
        assert(layer_id >= 0)
        parent_id = nf.map_to_parent_nid(i)
        assert F.array_equal(nf.layers[layer_id].data['h1'][local_nid], g.ndata['h1'][parent_id])

    for i in range(nf.number_of_edges()):
        flow_id, local_eid = nf.map_to_flow_eid(i)
        assert(flow_id >= 0)
        parent_id = nf.map_to_parent_eid(i)
        assert F.array_equal(nf.flows[flow_id].data['h2'][local_eid], g.edata['h2'][parent_id])


def test_basic():
    num_layers = 2
    g = generate_rand_graph(100, connect_more=True)
    nf = create_full_node_flow(g, num_layers)
    assert nf.number_of_nodes() == g.number_of_nodes() * (num_layers + 1)
    assert nf.number_of_edges() == g.number_of_edges() * num_layers
    assert nf.num_layers == num_layers + 1
    assert nf.layer_size(0) == g.number_of_nodes()
    assert nf.layer_size(1) == g.number_of_nodes()
    check_basic(g, nf)

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
    check_apply_nodes(create_full_node_flow)
    check_apply_nodes(create_mini_batch)


def check_apply_edges(create_node_flow):
    num_layers = 2
    for i in range(num_layers):
        g = generate_rand_graph(100)
        nf = create_node_flow(g, num_layers)
        nf.copy_from_parent()
        new_feats = F.randn((nf.flow_size(i), 5))
        def update_func(nodes):
            return {'h2' : new_feats}
        nf.apply_flow(i, update_func)
        assert F.array_equal(nf.flows[i].data['h2'], new_feats)


def test_apply_edges():
    check_apply_edges(create_full_node_flow)
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
        nf.flow_compute(i, fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                         lambda nodes: {'h' : nodes.data['t'] + 1})
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                     lambda nodes: {'h' : nodes.data['t'] + 1})
        assert F.array_equal(nf.layers[i + 1].data['h'], g.ndata['h'][nf.layer_parent_nid(i + 1)])

    # Test the computation when only a few nodes are active in a layer.
    g.ndata['h'] = g.ndata['h1']
    for i in range(num_layers):
        vs = nf.layer_nid(i+1)[0:4]
        nf.flow_compute(i, fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                        lambda nodes: {'h' : nodes.data['t'] + 1}, v=vs)
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                     lambda nodes: {'h' : nodes.data['t'] + 1})
        data1 = nf.layers[i + 1].data['h'][0:4]
        data2 = g.ndata['h'][nf.map_to_parent_nid(vs)]
        assert F.array_equal(data1, data2)


def test_flow_compute():
    check_flow_compute(create_full_node_flow)
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
    nf2.prop_flows(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                   lambda nodes: {'h' : nodes.data['t'] + 1})
    assert F.array_equal(nf2.layers[-1].data['h'], g.ndata['h'][nf2.layer_parent_nid(-1)])


def test_prop_flows():
    check_prop_flows(create_full_node_flow)
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
    for i in range(nf.num_flows):
        assert len(g.edata.keys()) == len(nf.flows[i].data.keys())
        for key in g.edata.keys():
            assert key in nf.flows[i].data.keys()
            assert F.array_equal(nf.flows[i].data[key], g.edata[key][nf.flow_parent_eid(i)])

    nf = create_mini_batch(g, num_layers)
    node_embed_names = [['h'], ['h1'], ['h']]
    edge_embed_names = [['h2'], ['h2']]
    nf.copy_from_parent(node_embed_names=node_embed_names, edge_embed_names=edge_embed_names)
    for i in range(nf.num_layers):
        assert len(node_embed_names[i]) == len(nf.layers[i].data.keys())
        for key in node_embed_names[i]:
            assert key in nf.layers[i].data.keys()
            assert F.array_equal(nf.layers[i].data[key], g.ndata[key][nf.layer_parent_nid(i)])
    for i in range(nf.num_flows):
        assert len(edge_embed_names[i]) == len(nf.flows[i].data.keys())
        for key in edge_embed_names[i]:
            assert key in nf.flows[i].data.keys()
            assert F.array_equal(nf.flows[i].data[key], g.edata[key][nf.flow_parent_eid(i)])

    nf = create_mini_batch(g, num_layers)
    g.ndata['h0'] = g.ndata['h'].copy()
    node_embed_names = [['h0'], [], []]
    nf.copy_from_parent(node_embed_names=node_embed_names, edge_embed_names=None)
    for i in range(num_layers):
        nf.flow_compute(i, fn.copy_src(src='h%d' % i, out='m'), fn.sum(msg='m', out='t'),
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
    g.ndata['h0'] = g.ndata['h'].copy()
    g.ndata['h1'] = g.ndata['h'].copy()
    g.ndata['h2'] = g.ndata['h'].copy()
    node_embed_names = [['h0'], ['h1'], ['h2']]
    nf.copy_from_parent(node_embed_names=node_embed_names, edge_embed_names=None)

    def msg_func(edge, ind):
        assert 'h%d' % ind in edge.src.keys()
        return {'m' : edge.src['h%d' % ind]}
    def reduce_func(node, ind):
        assert 'h%d' % (ind + 1) in node.data.keys()
        return {'h' : F.sum(node.mailbox['m'], 1) + node.data['h%d' % (ind + 1)]}

    for i in range(num_layers):
        nf.flow_compute(i, partial(msg_func, ind=i), partial(reduce_func, ind=i))


if __name__ == '__main__':
    test_basic()
    test_copy()
    test_apply_nodes()
    test_apply_edges()
    test_flow_compute()
    test_prop_flows()
