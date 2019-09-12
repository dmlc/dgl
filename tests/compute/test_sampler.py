import backend as F
import numpy as np
import scipy as sp
import dgl
from dgl import utils
from numpy.testing import assert_array_equal

np.random.seed(42)

def generate_rand_graph(n):
    arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
    return dgl.DGLGraph(arr, readonly=True)

def test_create_full():
    g = generate_rand_graph(100)
    full_nf = dgl.contrib.sampling.sampler.create_full_nodeflow(g, 5)
    assert full_nf.number_of_nodes() == g.number_of_nodes() * 6
    assert full_nf.number_of_edges() == g.number_of_edges() * 5

def test_1neighbor_sampler_all():
    g = generate_rand_graph(100)
    # In this case, NeighborSampling simply gets the neighborhood of a single vertex.
    for i, subg in enumerate(dgl.contrib.sampling.NeighborSampler(
            g, 1, g.number_of_nodes(), neighbor_type='in', num_workers=4)):
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
    for subg in dgl.contrib.sampling.NeighborSampler(g, 10, g.number_of_nodes(),
                                                     neighbor_type='in', num_workers=4):
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

def _test_layer_sampler(prefetch=False):
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

def test_layer_sampler():
    _test_layer_sampler()
    _test_layer_sampler(prefetch=True)

def test_nonuniform_neighbor_sampler():
    # Construct a graph with
    # (1) A path (0, 1, ..., 99) with weight 1
    # (2) A bunch of random edges with weight 0.
    edges = []
    for i in range(99):
        edges.append((i, i + 1))
    for i in range(1000):
        edge = (np.random.randint(100), np.random.randint(100))
        if edge not in edges:
            edges.append(edge)
    src, dst = zip(*edges)
    g = dgl.DGLGraph()
    g.add_nodes(100)
    g.add_edges(src, dst)
    g.readonly()

    g.edata['w'] = F.cat([
        F.ones((99,), F.float64, F.cpu()),
        F.zeros((len(edges) - 99,), F.float64, F.cpu())], 0)

    # Test 1-neighbor NodeFlow with 99 as target node.
    # The generated NodeFlow should only contain node i on layer i.
    sampler = dgl.contrib.sampling.NeighborSampler(
        g, 1, 1, 99, 'in', transition_prob='w', seed_nodes=[99])
    nf = next(iter(sampler))

    assert nf.num_layers == 100
    for i in range(nf.num_layers):
        assert nf.layer_size(i) == 1
        assert nf.layer_parent_nid(i)[0] == i

    # Test the reverse direction
    sampler = dgl.contrib.sampling.NeighborSampler(
        g, 1, 1, 99, 'out', transition_prob='w', seed_nodes=[0])
    nf = next(iter(sampler))

    assert nf.num_layers == 100
    for i in range(nf.num_layers):
        assert nf.layer_size(i) == 1
        assert nf.layer_parent_nid(i)[0] == 99 - i

def test_setseed():
    g = generate_rand_graph(100)

    nids = []

    dgl.random.seed(42)
    for subg in dgl.contrib.sampling.NeighborSampler(
            g, 5, 3, num_hops=2, neighbor_type='in', num_workers=1):
        nids.append(
            tuple(tuple(F.asnumpy(subg.layer_parent_nid(i))) for i in range(3)))

    # reinitialize
    dgl.random.seed(42)
    for i, subg in enumerate(dgl.contrib.sampling.NeighborSampler(
            g, 5, 3, num_hops=2, neighbor_type='in', num_workers=1)):
        item = tuple(tuple(F.asnumpy(subg.layer_parent_nid(i))) for i in range(3))
        assert item == nids[i]

    for i, subg in enumerate(dgl.contrib.sampling.NeighborSampler(
            g, 5, 3, num_hops=2, neighbor_type='in', num_workers=4)):
        pass

def check_negative_sampler(mode, exclude_positive):
    g = generate_rand_graph(100)

    pos_gsrc, pos_gdst, pos_geid = g.all_edges(form='all', order='eid')
    pos_map = {}
    for i in range(len(pos_geid)):
        pos_d = int(F.asnumpy(pos_gdst[i]))
        pos_e = int(F.asnumpy(pos_geid[i]))
        pos_map[(pos_d, pos_e)] = int(F.asnumpy(pos_gsrc[i]))

    EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
    neg_size = 10
    for pos_edges, neg_edges in EdgeSampler(g, 50,
                                            negative_mode=mode,
                                            neg_sample_size=neg_size,
                                            exclude_positive=exclude_positive):
        pos_nid = pos_edges.parent_nid
        pos_eid = pos_edges.parent_eid
        pos_lsrc, pos_ldst, pos_leid = pos_edges.all_edges(form='all', order='eid')
        pos_src = pos_nid[pos_lsrc]
        pos_dst = pos_nid[pos_ldst]
        pos_eid = pos_eid[pos_leid]
        assert_array_equal(F.asnumpy(pos_eid), F.asnumpy(g.edge_ids(pos_src, pos_dst)))

        neg_lsrc, neg_ldst, neg_leid = neg_edges.all_edges(form='all', order='eid')
        neg_nid = neg_edges.parent_nid
        neg_eid = neg_edges.parent_eid
        neg_src = neg_nid[neg_lsrc]
        neg_dst = neg_nid[neg_ldst]
        neg_eid = neg_eid[neg_leid]

        for i in range(len(neg_eid)):
            neg_d = int(F.asnumpy(neg_dst[i]))
            neg_e = int(F.asnumpy(neg_eid[i]))
            assert (neg_d, neg_e) in pos_map
            if exclude_positive:
                assert int(F.asnumpy(neg_src[i])) != pos_map[(neg_d, neg_e)]

def test_negative_sampler():
    check_negative_sampler('head', True)
    check_negative_sampler('PBG-head', False)


if __name__ == '__main__':
    test_create_full()
    test_1neighbor_sampler_all()
    test_10neighbor_sampler_all()
    test_1neighbor_sampler()
    test_10neighbor_sampler()
    test_layer_sampler()
    test_nonuniform_neighbor_sampler()
    test_setseed()
    test_negative_sampler()
