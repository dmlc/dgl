import os
os.environ['DGLBACKEND'] = 'mxnet'
import mxnet as mx
import numpy as np
import scipy as sp
import dgl
from dgl import utils

def generate_rand_graph(n, with_data=False):
    arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
    g = dgl.DGLGraph(arr, readonly=True)
    if with_data:
        g.ndata['h1'] = mx.nd.random.normal(shape=(g.number_of_nodes(), 10))
        g.ndata['h2'] = mx.nd.random.normal(shape=(g.number_of_nodes(), 9))
        g.edata['w'] = mx.nd.random.normal(shape=(g.number_of_edges(), 5))
    return g

def test_1neighbor_sampler_all():
    g = generate_rand_graph(100)
    # In this case, NeighborSampling simply gets the neighborhood of a single vertex.
    for subg, aux in dgl.contrib.sampling.NeighborSampler(g, 1, 100, neighbor_type='in',
                                                          num_workers=4, return_seed_id=True):
        seed_ids = aux['seeds']
        assert len(seed_ids) == 1
        src, dst, eid = g.in_edges(seed_ids, form='all')
        # Test if there is a self loop
        self_loop = mx.nd.sum(src == dst).asnumpy() == 1
        if self_loop:
            assert subg.number_of_nodes() == len(src)
        else:
            assert subg.number_of_nodes() == len(src) + 1
        assert subg.number_of_edges() >= len(src)

        child_ids = subg.map_to_subgraph_nid(seed_ids)
        child_src, child_dst, child_eid = subg.in_edges(child_ids, form='all')

        child_src1 = subg.map_to_subgraph_nid(src)
        assert mx.nd.sum(child_src1 == child_src).asnumpy() == len(src)

def is_sorted(arr):
    return np.sum(np.sort(arr) == arr) == len(arr)

def verify_subgraph(g, subg, seed_id):
    src, dst, eid = g.in_edges(seed_id, form='all')
    child_id = subg.map_to_subgraph_nid(seed_id)
    child_src, child_dst, child_eid = subg.in_edges(child_id, form='all')
    child_src = child_src.asnumpy()
    # We don't allow duplicate elements in the neighbor list.
    assert(len(np.unique(child_src)) == len(child_src))
    # The neighbor list also needs to be sorted.
    assert(is_sorted(child_src))

    child_src1 = subg.map_to_subgraph_nid(src).asnumpy()
    child_src1 = child_src1[child_src1 >= 0]
    for i in child_src:
        assert i in child_src1

def test_1neighbor_sampler():
    g = generate_rand_graph(100)
    # In this case, NeighborSampling simply gets the neighborhood of a single vertex.
    for subg, aux in dgl.contrib.sampling.NeighborSampler(g, 1, 5, neighbor_type='in',
                                                          num_workers=4, return_seed_id=True):
        seed_ids = aux['seeds']
        assert len(seed_ids) == 1
        assert subg.number_of_nodes() <= 6
        assert subg.number_of_edges() <= 5
        verify_subgraph(g, subg, seed_ids)

def test_prefetch_neighbor_sampler():
    g = generate_rand_graph(100)
    # In this case, NeighborSampling simply gets the neighborhood of a single vertex.
    for subg, aux in dgl.contrib.sampling.NeighborSampler(g, 1, 5, neighbor_type='in',
                                                          num_workers=4, return_seed_id=True, prefetch=True):
        seed_ids = aux['seeds']
        assert len(seed_ids) == 1
        assert subg.number_of_nodes() <= 6
        assert subg.number_of_edges() <= 5
        verify_subgraph(g, subg, seed_ids)

def test_10neighbor_sampler_all():
    g = generate_rand_graph(100)
    # In this case, NeighborSampling simply gets the neighborhood of a single vertex.
    for subg, aux in dgl.contrib.sampling.NeighborSampler(g, 10, 100, neighbor_type='in',
                                                          num_workers=4, return_seed_id=True):
        seed_ids = aux['seeds']
        src, dst, eid = g.in_edges(seed_ids, form='all')

        child_ids = subg.map_to_subgraph_nid(seed_ids)
        child_src, child_dst, child_eid = subg.in_edges(child_ids, form='all')

        child_src1 = subg.map_to_subgraph_nid(src)
        assert mx.nd.sum(child_src1 == child_src).asnumpy() == len(src)

def check_10neighbor_sampler(g, seeds):
    # In this case, NeighborSampling simply gets the neighborhood of a single vertex.
    for subg, aux in dgl.contrib.sampling.NeighborSampler(g, 10, 5, neighbor_type='in',
                                                          num_workers=4, seed_nodes=seeds,
                                                          return_seed_id=True):
        seed_ids = aux['seeds']
        assert subg.number_of_nodes() <= 6 * len(seed_ids)
        assert subg.number_of_edges() <= 5 * len(seed_ids)
        for seed_id in seed_ids:
            verify_subgraph(g, subg, seed_id)

def test_10neighbor_sampler():
    g = generate_rand_graph(100)
    check_10neighbor_sampler(g, None)
    check_10neighbor_sampler(g, seeds=np.unique(np.random.randint(0, g.number_of_nodes(),
                                                                  size=int(g.number_of_nodes() / 10))))


def test_cached_sampler():
    g = generate_rand_graph(100, True)
    batch_size = 10
    sampler = dgl.contrib.sampling.NeighborSampler(g, batch_size, 5, neighbor_type='in',
                                                   num_workers=4, return_seed_id=True,
                                                   cache_nodes=range(0, 100, 3))
    num_iters = 0
    for subg, aux in sampler:
        assert subg._vertex_cache is not None
        subg.copy_from_parent()
        assert np.all(subg.ndata['h1'].asnumpy() == g.ndata['h1'][subg.parent_nid].asnumpy())
        assert np.all(subg.ndata['h2'].asnumpy() == g.ndata['h2'][subg.parent_nid].asnumpy())
        assert np.all(subg.edata['w'].asnumpy() == g.edata['w'][subg.parent_eid].asnumpy())
        num_iters += 1
    assert num_iters == g.number_of_nodes() / batch_size

    g.ndata['h2'] = mx.nd.random.normal(shape=(g.number_of_nodes(), 9))
    # We refresh everything in the cache.
    sampler.reset()
    for subg, aux in sampler:
        assert subg._vertex_cache is not None
        subg.copy_from_parent()
        assert np.all(subg.ndata['h1'].asnumpy() == g.ndata['h1'][subg.parent_nid].asnumpy())
        assert np.all(subg.ndata['h2'].asnumpy() == g.ndata['h2'][subg.parent_nid].asnumpy())
        assert np.all(subg.edata['w'].asnumpy() == g.edata['w'][subg.parent_eid].asnumpy())
    assert num_iters == g.number_of_nodes() / batch_size

    g.ndata['h2'] = mx.nd.random.normal(shape=(g.number_of_nodes(), 9))
    # We refresh h2
    sampler.reset(pin_cache_keys=['h1'])
    for subg, aux in sampler:
        assert subg._vertex_cache is not None
        subg.copy_from_parent()
        assert np.all(subg.ndata['h1'].asnumpy() == g.ndata['h1'][subg.parent_nid].asnumpy())
        assert np.all(subg.ndata['h2'].asnumpy() == g.ndata['h2'][subg.parent_nid].asnumpy())
        assert np.all(subg.edata['w'].asnumpy() == g.edata['w'][subg.parent_eid].asnumpy())
    assert num_iters == g.number_of_nodes() / batch_size

    g.ndata['h2'] = mx.nd.random.normal(shape=(g.number_of_nodes(), 9))
    # We update h2 but don't refresh h2, then h2 in subgraph shouldn't match h2
    # in the original graph.
    sampler.reset(pin_cache_keys=['h1', 'h2'])
    for subg, aux in sampler:
        assert subg._vertex_cache is not None
        subg.copy_from_parent()
        assert np.all(subg.ndata['h1'].asnumpy() == g.ndata['h1'][subg.parent_nid].asnumpy())
        assert np.any(subg.ndata['h2'].asnumpy() != g.ndata['h2'][subg.parent_nid].asnumpy())
        assert np.all(subg.edata['w'].asnumpy() == g.edata['w'][subg.parent_eid].asnumpy())
    assert num_iters == g.number_of_nodes() / batch_size


if __name__ == '__main__':
    test_1neighbor_sampler_all()
    test_10neighbor_sampler_all()
    test_1neighbor_sampler()
    test_10neighbor_sampler()
    test_cached_sampler()
