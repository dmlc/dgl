import backend as F
import numpy as np
import scipy as sp
import dgl
from dgl import utils

def generate_rand_graph(n):
    arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
    return dgl.DGLGraph(arr, readonly=True)

def test_create_full():
    g = generate_rand_graph(100)
    full_nf = dgl.contrib.sampling.sampler.create_full_nodeflow(g, 5)
    assert full_nf.number_of_nodes() == 600
    assert full_nf.number_of_edges() == g.number_of_edges() * 5

def test_1neighbor_sampler_all():
    g = generate_rand_graph(100)
    # In this case, NeighborSampling simply gets the neighborhood of a single vertex.
    for i, subg in enumerate(dgl.contrib.sampling.NeighborSampler(
            g, 1, 100, neighbor_type='in', num_workers=4)):
        seed_ids = subg.layer_parent_nid(-1)
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
    for subg in dgl.contrib.sampling.NeighborSampler(g, 1, 5, neighbor_type='in',
                                                     num_workers=4):
        seed_ids = subg.layer_parent_nid(-1)
        assert len(seed_ids) == 1
        assert subg.number_of_nodes() <= 6
        assert subg.number_of_edges() <= 5
        verify_subgraph(g, subg, seed_ids)

def test_prefetch_neighbor_sampler():
    g = generate_rand_graph(100)
    # In this case, NeighborSampling simply gets the neighborhood of a single vertex.
    for subg in dgl.contrib.sampling.NeighborSampler(g, 1, 5, neighbor_type='in',
                                                     num_workers=4, prefetch=True):
        seed_ids = subg.layer_parent_nid(-1)
        assert len(seed_ids) == 1
        assert subg.number_of_nodes() <= 6
        assert subg.number_of_edges() <= 5
        verify_subgraph(g, subg, seed_ids)

def test_10neighbor_sampler_all():
    g = generate_rand_graph(100)
    # In this case, NeighborSampling simply gets the neighborhood of a single vertex.
    for subg in dgl.contrib.sampling.NeighborSampler(g, 10, 100, neighbor_type='in',
                                                     num_workers=4):
        seed_ids = subg.layer_parent_nid(-1)
        assert F.array_equal(seed_ids, subg.map_to_parent_nid(subg.layer_nid(-1)))

        src, dst, eid = g.in_edges(seed_ids, form='all')
        child_src, child_dst, child_eid = subg.in_edges(subg.layer_nid(-1), form='all')
        src1 = subg.map_to_parent_nid(child_src)
        assert F.array_equal(src1, src)

def check_10neighbor_sampler(g, seeds):
    # In this case, NeighborSampling simply gets the neighborhood of a single vertex.
    for subg in dgl.contrib.sampling.NeighborSampler(g, 10, 5, neighbor_type='in',
                                                     num_workers=4, seed_nodes=seeds):
        seed_ids = subg.layer_parent_nid(-1)
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
    n_batches = 5
    batch_size = 50
    seed_batches = [np.sort(np.random.choice(F.asnumpy(nid), batch_size, replace=False))
                    for i in range(n_batches)]
    seed_nodes = np.hstack(seed_batches)
    layer_sizes = [50] * 3
    LayerSampler = getattr(dgl.contrib.sampling, 'LayerSampler')
    sampler = LayerSampler(g, batch_size, layer_sizes, 'in',
                           seed_nodes=seed_nodes, num_workers=4, prefetch=prefetch)
    for sub_g in sampler:
        assert all(sub_g.layer_size(i) < size for i, size in enumerate(layer_sizes))
        sub_nid = F.arange(0, sub_g.number_of_nodes())
        assert all(np.all(np.isin(F.asnumpy(sub_g.layer_nid(i)), F.asnumpy(sub_nid)))
                   for i in range(sub_g.num_layers))
        assert np.all(np.isin(F.asnumpy(sub_g.map_to_parent_nid(sub_nid)),
                              F.asnumpy(nid)))
        sub_eid = F.arange(0, sub_g.number_of_edges())
        assert np.all(np.isin(F.asnumpy(sub_g.map_to_parent_eid(sub_eid)),
                              F.asnumpy(eid)))
        assert any(np.all(np.sort(F.asnumpy(sub_g.layer_parent_nid(-1))) == seed_batch)
                   for seed_batch in seed_batches)

        sub_src, sub_dst = sub_g.all_edges(order='eid')
        for i in range(sub_g.num_blocks):
            block_eid = sub_g.block_eid(i)
            block_src = sub_g.map_to_parent_nid(sub_src[block_eid])
            block_dst = sub_g.map_to_parent_nid(sub_dst[block_eid])

            block_parent_eid = sub_g.block_parent_eid(i)
            block_parent_src = src[block_parent_eid]
            block_parent_dst = dst[block_parent_eid]

            assert np.all(F.asnumpy(block_src == block_parent_src))

        n_layers = sub_g.num_layers
        sub_n = sub_g.number_of_nodes()
        assert sum(F.shape(sub_g.layer_nid(i))[0] for i in range(n_layers)) == sub_n
        n_blocks = sub_g.num_blocks
        sub_m = sub_g.number_of_edges()
        assert sum(F.shape(sub_g.block_eid(i))[0] for i in range(n_blocks)) == sub_m

if __name__ == '__main__':
    test_create_full()
    test_1neighbor_sampler_all()
    test_10neighbor_sampler_all()
    test_1neighbor_sampler()
    test_10neighbor_sampler()
    test_layer_sampler()
    test_layer_sampler(prefetch=True)
