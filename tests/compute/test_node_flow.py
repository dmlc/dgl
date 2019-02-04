import backend as F
import numpy as np
import scipy as sp
import dgl
from dgl.node_flow import create_full_node_flow
from dgl import utils
import dgl.function as fn

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


def create_mini_batch(g, seed_ids, num_hops):
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
    seed_ids = np.array([0, 1, 2, 3])
    nf = create_mini_batch(g, seed_ids, num_layers)
    assert nf.num_layers == num_layers + 1
    assert nf.layer_size(-1) == len(seed_ids)
    check_basic(g, nf)


def test_apply_nodes():
    num_layers = 2
    for i in range(num_layers):
        g = generate_rand_graph(100)
        nf = create_full_node_flow(g, num_layers)
        nf.copy_from_parent()
        new_feats = F.randn((nf.layer_size(i), 5))
        def update_func(nodes):
            return {'h1' : new_feats}
        nf.apply_layer(i, update_func)
        assert F.array_equal(nf.layers[i].data['h1'], new_feats)


def test_apply_edges():
    num_layers = 2
    for i in range(num_layers):
        g = generate_rand_graph(100)
        nf = create_full_node_flow(g, num_layers)
        nf.copy_from_parent()
        new_feats = F.randn((nf.flow_size(i), 5))
        def update_func(nodes):
            return {'h2' : new_feats}
        nf.apply_flow(i, update_func)
        assert F.array_equal(nf.flows[i].data['h2'], new_feats)


def test_flow_compute():
    num_layers = 2
    g = generate_rand_graph(100)
    nf1 = create_full_node_flow(g, num_layers)
    nf1.copy_from_parent()
    nf2 = create_full_node_flow(g, num_layers)
    nf2.copy_from_parent()
    g.ndata['h'] = g.ndata['h1']
    nf1.layers[0].data['h'] = nf1.layers[0].data['h1']
    # Test the computation on a layer at a time.
    for i in range(num_layers):
        nf1.flow_compute(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                         lambda nodes: {'h' : nodes.data['t'] + 1}, range=i)
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
                     lambda nodes: {'h' : nodes.data['t'] + 1})
        assert F.array_equal(nf1.layers[i + 1].data['h'], g.ndata['h'])
        
    # Test the computation on all layers.
    #nf2.flow_compute(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='t'),
    #                 lambda nodes: {'h' : nodes.data['t'] + 1})
    #assert F.array_equal(nf2.layers[-1].data['h'], g.ndata['h'])


if __name__ == '__main__':
    test_basic()
    test_apply_nodes()
    test_apply_edges()
    test_flow_compute()
