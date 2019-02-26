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

        assert seed_ids == subg.layer_parent_nid(-1)
        child_src, child_dst, child_eid = subg.in_edges(subg.layer_nid(-1), form='all')
        assert F.array_equal(child_src, subg.layer_nid(0))

        src1 = subg.map_to_parent_nid(child_src)
        assert F.array_equal(src1, src)

def is_sorted(arr):
    return np.sum(np.sort(arr) == arr, 0) == len(arr)

def verify_subgraph(g, subg, seed_id):
    seed_id = F.asnumpy(seed_id)
    seeds = F.asnumpy(subg.map_to_parent_nid(subg.layer_nid(-1)))
    assert seed_id in seeds
    child_seed = F.asnumpy(subg.layer_nid(-1))[seeds == seed_id]
    src, dst, eid = g.in_edges(seed_id, form='all')
    child_src, child_dst, child_eid = subg.in_edges(child_seed, form='all')

    child_src = F.asnumpy(child_src)
    # We don't allow duplicate elements in the neighbor list.
    assert(len(np.unique(child_src)) == len(child_src))
    # The neighbor list also needs to be sorted.
    assert(is_sorted(child_src))

    # a neighbor in the subgraph must also exist in parent graph.
    for i in subg.map_to_parent_nid(child_src):
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
        assert F.array_equal(seed_ids, subg.map_to_parent_nid(subg.layer_nid(-1)))

        src, dst, eid = g.in_edges(seed_ids, form='all')
        child_src, child_dst, child_eid = subg.in_edges(subg.layer_nid(-1), form='all')
        src1 = subg.map_to_parent_nid(child_src)
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

def test_layer_sampler(prefetch=False):
    g = generate_rand_graph(100)
    nid = g.nodes()
    src, dst, eid = g.all_edges(form='all', order='eid')
    seed_nodes = np.sort(np.random.choice(F.asnumpy(nid), 50, replace=False))
    layer_sizes = [50] * 2
    LayerSampler = getattr(dgl.contrib.sampling, 'LayerSampler')
    sampler = LayerSampler(g, len(seed_nodes), layer_sizes, 'in',
                           seed_nodes=seed_nodes, prefetch=prefetch)
    for sub_g, ret in sampler:
        assert all(sub_g.layer_size(i) < size for i, size in enumerate(layer_sizes))
        sub_nid = np.arange(0, sub_g.number_of_nodes())
        assert all(np.all(np.isin(F.asnumpy(sub_g.layer_nid(i)), sub_nid))
                   for i in range(sub_g.num_layers))
        assert np.all(np.isin(F.asnumpy(sub_g.map_to_parent_nid(sub_nid)),
                              F.asnumpy(nid)))
        sub_eid = F.arange(0, sub_g.number_of_edges())
        assert np.all(np.isin(F.asnumpy(sub_g.map_to_parent_eid(sub_eid)),
                              F.asnumpy(eid)))
        assert np.all(seed_nodes == np.sort(F.asnumpy(sub_g.layer_parent_nid(-1))))

        sub_src, sub_dst = sub_g.all_edges(order='eid')
        for i in range(sub_g.num_blocks):
            block_size = sub_g.block_size(i)

            block_eid = sub_g.block_eid(i)
            block_src = sub_g.map_to_parent_nid(sub_src[block_eid])
            block_dst = sub_g.map_to_parent_nid(sub_dst[block_eid])

            block_parent_eid = sub_g.block_parent_eid(i)
            block_parent_src = src[block_parent_eid]
            block_parent_dst = dst[block_parent_eid]

            assert np.all(block_src == block_parent_src)

def test_random_walk():
    edge_list = [(0, 1), (1, 2), (2, 3), (3, 4),
                 (4, 3), (3, 2), (2, 1), (1, 0)]
    seeds = [0, 1]
    n_traces = 3
    n_hops = 4

    g = dgl.DGLGraph(edge_list, readonly=True)
    traces = dgl.contrib.sampling.random_walk(g, seeds, n_traces, n_hops)
    traces = F.zerocopy_to_numpy(traces)

    assert traces.shape == (len(seeds), n_traces, n_hops + 1)

    for i, seed in enumerate(seeds):
        assert (traces[i, :, 0] == seeds[i]).all()

    trace_diff = np.diff(traces, axis=-1)
    # only nodes with adjacent IDs are connected
    assert (np.abs(trace_diff) == 1).all()

if __name__ == '__main__':
    test_1neighbor_sampler_all()
    test_10neighbor_sampler_all()
    test_1neighbor_sampler()
    test_10neighbor_sampler()
    test_layer_sampler()
    test_layer_sampler(prefetch=True)
    test_random_walk()
