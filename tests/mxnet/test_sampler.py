import os
os.environ['DGLBACKEND'] = 'mxnet'
import mxnet as mx
import numpy as np
import scipy as sp
import dgl
from dgl import utils

def generate_rand_graph(n):
    arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
    return dgl.DGLGraph(arr, readonly=True)

def test_1neighbor_sampler_all():
    g = generate_rand_graph(100)
    # In this case, NeighborSampling simply gets the neighborhood of a single vertex.
    for subg, seed_ids in dgl.sampling.NeighborSampler(g, 1, 'ALL', neighbor_type='in'):
        assert len(seed_ids) == 1
        src, dst, eid = g._graph.in_edges(seed_ids)
        # Test if there is a self loop
        self_loop = mx.nd.sum(src.tousertensor() == dst.tousertensor()).asnumpy() == 1
        if self_loop:
            assert subg.number_of_nodes() == len(src)
        else:
            assert subg.number_of_nodes() == len(src) + 1
        assert subg.number_of_edges() >= len(src)

        child_ids = subg.map_to_subgraph_nid(seed_ids)
        child_src, child_dst, child_eid = subg._graph.in_edges(child_ids)

        child_src1 = subg.map_to_subgraph_nid(src)
        assert mx.nd.sum(child_src1.tousertensor() == child_src.tousertensor()).asnumpy() == len(src)

def is_sorted(arr):
    return np.sum(np.sort(arr) == arr) == len(arr)

def verify_subgraph(g, subg, seed_id):
    seed_id = utils.toindex(seed_id)
    src, dst, eid = g._graph.in_edges(seed_id)
    child_id = subg.map_to_subgraph_nid(seed_id)
    child_src, child_dst, child_eid = subg._graph.in_edges(child_id)
    child_src = child_src.tousertensor().asnumpy()
    # We don't allow duplicate elements in the neighbor list.
    assert(len(np.unique(child_src)) == len(child_src))
    # The neighbor list also needs to be sorted.
    assert(is_sorted(child_src))

    child_src1 = subg.map_to_subgraph_nid(src).tousertensor().asnumpy()
    child_src1 = child_src1[child_src1 >= 0]
    for i in child_src:
        assert i in child_src1

def test_1neighbor_sampler():
    g = generate_rand_graph(100)
    # In this case, NeighborSampling simply gets the neighborhood of a single vertex.
    for subg, seed_ids in dgl.sampling.NeighborSampler(g, 1, 5, neighbor_type='in'):
        assert len(seed_ids) == 1
        assert subg.number_of_nodes() <= 6
        assert subg.number_of_edges() <= 5
        verify_subgraph(g, subg, seed_ids)

def test_10neighbor_sampler_all():
    g = generate_rand_graph(100)
    # In this case, NeighborSampling simply gets the neighborhood of a single vertex.
    for subg, seed_ids in dgl.sampling.NeighborSampler(g, 10, 'ALL', neighbor_type='in'):
        src, dst, eid = g._graph.in_edges(seed_ids)

        child_ids = subg.map_to_subgraph_nid(seed_ids)
        child_src, child_dst, child_eid = subg._graph.in_edges(child_ids)

        child_src1 = subg.map_to_subgraph_nid(src)
        assert mx.nd.sum(child_src1.tousertensor() == child_src.tousertensor()).asnumpy() == len(src)

def test_10neighbor_sampler():
    g = generate_rand_graph(100)
    # In this case, NeighborSampling simply gets the neighborhood of a single vertex.
    for subg, seed_ids in dgl.sampling.NeighborSampler(g, 10, 5, neighbor_type='in'):
        assert subg.number_of_nodes() <= 6 * len(seed_ids)
        assert subg.number_of_edges() <= 5 * len(seed_ids)
        for seed_id in seed_ids.tousertensor():
            verify_subgraph(g, subg, seed_id)

if __name__ == '__main__':
    test_1neighbor_sampler()
    test_10neighbor_sampler()
