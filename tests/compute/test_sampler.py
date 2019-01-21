import backend as F
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
    for subg, aux in dgl.contrib.sampling.NeighborSampler(g, 1, 100, neighbor_type='in',
                                                          num_workers=4, return_seed_id=True):
        seed_ids = aux['seeds']
        assert len(seed_ids) == 1
        src, dst, eid = g.in_edges(seed_ids, form='all')
        assert subg.number_of_nodes() == len(src) + 1
        assert subg.number_of_edges() == len(src)

        assert F.array_equal(seed_ids, subg.layer_parent_nid(0))
        child_src, child_dst, child_eid = subg.in_edges(subg.layer_nid(0), form='all')
        assert F.array_equal(child_src, subg.layer_nid(1))

        src1 = subg.parent_nid[child_src]
        assert F.array_equal(src1, src)

def is_sorted(arr):
    return np.sum(np.sort(arr) == arr, 0) == len(arr)

def verify_subgraph(g, subg, seed_id):
    seed_id = F.asnumpy(seed_id)
    seeds = F.asnumpy(subg.parent_nid[subg.layer_nid(0)])
    assert seed_id in seeds
    child_seed = F.asnumpy(subg.layer_nid(0))[seeds == seed_id]
    src, dst, eid = g.in_edges(seed_id, form='all')
    child_src, child_dst, child_eid = subg.in_edges(child_seed, form='all')

    child_src = F.asnumpy(child_src)
    # We don't allow duplicate elements in the neighbor list.
    assert(len(np.unique(child_src)) == len(child_src))
    # The neighbor list also needs to be sorted.
    assert(is_sorted(child_src))

    # a neighbor in the subgraph must also exist in parent graph.
    for i in subg.parent_nid[child_src]:
        assert i in src

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
        assert F.array_equal(seed_ids, subg.parent_nid[subg.layer_nid(0)])

        src, dst, eid = g.in_edges(seed_ids, form='all')

        child_src, child_dst, child_eid = subg.in_edges(subg.layer_nid(0), form='all')

        src1 = subg.parent_nid[child_src]
        assert F.array_equal(src1, src)

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

if __name__ == '__main__':
    test_1neighbor_sampler_all()
    test_10neighbor_sampler_all()
    test_1neighbor_sampler()
    test_10neighbor_sampler()
