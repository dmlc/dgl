import backend as F
import numpy as np
import scipy as sp
import dgl
from dgl import utils
import unittest
from numpy.testing import assert_array_equal

np.random.seed(42)

def generate_rand_graph(n):
    arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
    return dgl.DGLGraphStale(arr, readonly=True)

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
    src = F.asnumpy(src)
    for i in subg.map_to_parent_nid(child_src):
        assert F.asnumpy(i) in src

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
            block_src = sub_g.map_to_parent_nid(F.gather_row(sub_src, block_eid))
            block_dst = sub_g.map_to_parent_nid(F.gather_row(sub_dst, block_eid))

            block_parent_eid = sub_g.block_parent_eid(i)
            block_parent_src = F.gather_row(src, block_parent_eid)
            block_parent_dst = F.gather_row(dst, block_parent_eid)

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

@unittest.skipIf(dgl.backend.backend_name == "tensorflow", reason="Error occured when multiprocessing")
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
    g = dgl.DGLGraphStale()
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
        assert F.asnumpy(nf.layer_parent_nid(i)[0]) == i

    # Test the reverse direction
    sampler = dgl.contrib.sampling.NeighborSampler(
        g, 1, 1, 99, 'out', transition_prob='w', seed_nodes=[0])
    nf = next(iter(sampler))

    assert nf.num_layers == 100
    for i in range(nf.num_layers):
        assert nf.layer_size(i) == 1
        assert F.asnumpy(nf.layer_parent_nid(i)[0]) == 99 - i

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

def check_head_tail(g):
    lsrc, ldst, leid = g.all_edges(form='all', order='eid')

    lsrc = np.unique(F.asnumpy(lsrc))
    head_nid = np.unique(F.asnumpy(g.head_nid))
    np.testing.assert_equal(lsrc, head_nid)

    ldst = np.unique(F.asnumpy(ldst))
    tail_nid = np.unique(F.asnumpy(g.tail_nid))
    np.testing.assert_equal(tail_nid, ldst)


def check_negative_sampler(mode, exclude_positive, neg_size):
    g = generate_rand_graph(100)
    num_edges = g.number_of_edges()
    etype = np.random.randint(0, 10, size=g.number_of_edges(), dtype=np.int64)
    g.edata['etype'] = F.copy_to(F.tensor(etype), F.cpu())

    pos_gsrc, pos_gdst, pos_geid = g.all_edges(form='all', order='eid')
    pos_map = {}
    for i in range(len(pos_geid)):
        pos_d = int(F.asnumpy(pos_gdst[i]))
        pos_e = int(F.asnumpy(pos_geid[i]))
        pos_map[(pos_d, pos_e)] = int(F.asnumpy(pos_gsrc[i]))

    EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
    # Test the homogeneous graph.

    batch_size = 50
    total_samples = 0
    for pos_edges, neg_edges in EdgeSampler(g, batch_size,
                                            negative_mode=mode,
                                            reset=False,
                                            neg_sample_size=neg_size,
                                            exclude_positive=exclude_positive,
                                            return_false_neg=True):
        pos_lsrc, pos_ldst, pos_leid = pos_edges.all_edges(form='all', order='eid')
        assert_array_equal(F.asnumpy(F.gather_row(pos_edges.parent_eid, pos_leid)),
                           F.asnumpy(g.edge_ids(F.gather_row(pos_edges.parent_nid, pos_lsrc),
                                                F.gather_row(pos_edges.parent_nid, pos_ldst))))

        neg_lsrc, neg_ldst, neg_leid = neg_edges.all_edges(form='all', order='eid')

        neg_src = F.gather_row(neg_edges.parent_nid, neg_lsrc)
        neg_dst = F.gather_row(neg_edges.parent_nid, neg_ldst)
        neg_eid = F.gather_row(neg_edges.parent_eid, neg_leid)
        for i in range(len(neg_eid)):
            neg_d = int(F.asnumpy(neg_dst)[i])
            neg_e = int(F.asnumpy(neg_eid)[i])
            assert (neg_d, neg_e) in pos_map
            if exclude_positive:
                assert int(F.asnumpy(neg_src[i])) != pos_map[(neg_d, neg_e)]

        check_head_tail(neg_edges)
        pos_tails = F.gather_row(pos_edges.parent_nid, pos_edges.tail_nid)
        neg_tails = F.gather_row(neg_edges.parent_nid, neg_edges.tail_nid)
        pos_tails = np.sort(F.asnumpy(pos_tails))
        neg_tails = np.sort(F.asnumpy(neg_tails))
        np.testing.assert_equal(pos_tails, neg_tails)

        exist = neg_edges.edata['false_neg']
        if exclude_positive:
            assert np.sum(F.asnumpy(exist) == 0) == len(exist)
        else:
            assert F.array_equal(g.has_edges_between(neg_src, neg_dst), exist)
        total_samples += batch_size
    assert total_samples <= num_edges

    # check replacement = True
    # with reset = False (default setting)
    total_samples = 0
    for pos_edges, neg_edges in EdgeSampler(g, batch_size,
                                            replacement=True,
                                            reset=False,
                                            negative_mode=mode,
                                            neg_sample_size=neg_size,
                                            exclude_positive=exclude_positive,
                                            return_false_neg=True):
        _, _, pos_leid = pos_edges.all_edges(form='all', order='eid')
        assert len(pos_leid) == batch_size
        total_samples += len(pos_leid)
    assert total_samples == num_edges

    # check replacement = False
    # with reset = False (default setting)
    total_samples = 0
    for pos_edges, neg_edges in EdgeSampler(g, batch_size,
                                            replacement=False,
                                            reset=False,
                                            negative_mode=mode,
                                            neg_sample_size=neg_size,
                                            exclude_positive=exclude_positive,
                                            return_false_neg=True):
        _, _, pos_leid = pos_edges.all_edges(form='all', order='eid')
        assert len(pos_leid) == batch_size
        total_samples += len(pos_leid)
    assert total_samples == num_edges

    # check replacement = True
    # with reset = True
    total_samples = 0
    max_samples = 2 * num_edges
    for pos_edges, neg_edges in EdgeSampler(g, batch_size,
                                            replacement=True,
                                            reset=True,
                                            negative_mode=mode,
                                            neg_sample_size=neg_size,
                                            exclude_positive=exclude_positive,
                                            return_false_neg=True):
        _, _, pos_leid = pos_edges.all_edges(form='all', order='eid')
        assert len(pos_leid) <= batch_size
        total_samples += len(pos_leid)
        if (total_samples >= max_samples):
            break
    assert total_samples >= max_samples

    # check replacement = False
    # with reset = True
    total_samples = 0
    max_samples = 2 * num_edges
    for pos_edges, neg_edges in EdgeSampler(g, batch_size,
                                            replacement=False,
                                            reset=True,
                                            negative_mode=mode,
                                            neg_sample_size=neg_size,
                                            exclude_positive=exclude_positive,
                                            return_false_neg=True):
        _, _, pos_leid = pos_edges.all_edges(form='all', order='eid')
        assert len(pos_leid) <= batch_size
        total_samples += len(pos_leid)
        if (total_samples >= max_samples):
            break
    assert total_samples >= max_samples

    # Test the knowledge graph.
    total_samples = 0
    for _, neg_edges in EdgeSampler(g, batch_size,
                                    negative_mode=mode,
                                    reset=False,
                                    neg_sample_size=neg_size,
                                    exclude_positive=exclude_positive,
                                    relations=g.edata['etype'],
                                    return_false_neg=True):
        neg_lsrc, neg_ldst, neg_leid = neg_edges.all_edges(form='all', order='eid')
        neg_src = F.gather_row(neg_edges.parent_nid, neg_lsrc)
        neg_dst = F.gather_row(neg_edges.parent_nid, neg_ldst)
        neg_eid = F.gather_row(neg_edges.parent_eid, neg_leid)
        exists = neg_edges.edata['false_neg']
        neg_edges.edata['etype'] = F.gather_row(g.edata['etype'], neg_eid)
        for i in range(len(neg_eid)):
            u, v = F.asnumpy(neg_src[i]), F.asnumpy(neg_dst[i])
            if g.has_edge_between(u, v):
                eid = g.edge_ids(u, v)
                etype = g.edata['etype'][eid]
                exist = neg_edges.edata['etype'][i] == etype
                assert F.asnumpy(exists[i]) == F.asnumpy(exist)
        total_samples += batch_size
    assert total_samples <= num_edges

def check_weighted_negative_sampler(mode, exclude_positive, neg_size):
    g = generate_rand_graph(100)
    num_edges = g.number_of_edges()
    num_nodes = g.number_of_nodes()
    edge_weight = F.copy_to(F.tensor(np.full((num_edges,), 1, dtype=np.float32)), F.cpu())
    node_weight = F.copy_to(F.tensor(np.full((num_nodes,), 1, dtype=np.float32)), F.cpu())
    etype = np.random.randint(0, 10, size=num_edges, dtype=np.int64)
    g.edata['etype'] = F.copy_to(F.tensor(etype), F.cpu())

    pos_gsrc, pos_gdst, pos_geid = g.all_edges(form='all', order='eid')
    pos_map = {}
    for i in range(len(pos_geid)):
        pos_d = int(F.asnumpy(pos_gdst[i]))
        pos_e = int(F.asnumpy(pos_geid[i]))
        pos_map[(pos_d, pos_e)] = int(F.asnumpy(pos_gsrc[i]))
    EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')

    # Correctness check
    # Test the homogeneous graph.
    batch_size = 50
    # Test the knowledge graph with edge weight provied.
    total_samples = 0
    for pos_edges, neg_edges in EdgeSampler(g, batch_size,
                                            reset=False,
                                            edge_weight=edge_weight,
                                            negative_mode=mode,
                                            neg_sample_size=neg_size,
                                            exclude_positive=exclude_positive,
                                            return_false_neg=True):
        pos_lsrc, pos_ldst, pos_leid = pos_edges.all_edges(form='all', order='eid')
        assert_array_equal(F.asnumpy(F.gather_row(pos_edges.parent_eid, pos_leid)),
                           F.asnumpy(g.edge_ids(F.gather_row(pos_edges.parent_nid, pos_lsrc),
                                                F.gather_row(pos_edges.parent_nid, pos_ldst))))
        neg_lsrc, neg_ldst, neg_leid = neg_edges.all_edges(form='all', order='eid')

        neg_src = F.gather_row(neg_edges.parent_nid, neg_lsrc)
        neg_dst = F.gather_row(neg_edges.parent_nid, neg_ldst)
        neg_eid = F.gather_row(neg_edges.parent_eid, neg_leid)
        for i in range(len(neg_eid)):
            neg_d = int(F.asnumpy(neg_dst[i]))
            neg_e = int(F.asnumpy(neg_eid[i]))
            assert (neg_d, neg_e) in pos_map
            if exclude_positive:
                assert int(F.asnumpy(neg_src[i])) != pos_map[(neg_d, neg_e)]

        check_head_tail(neg_edges)
        pos_tails = F.gather_row(pos_edges.parent_nid, pos_edges.tail_nid)
        neg_tails = F.gather_row(neg_edges.parent_nid, neg_edges.tail_nid)
        pos_tails = np.sort(F.asnumpy(pos_tails))
        neg_tails = np.sort(F.asnumpy(neg_tails))
        np.testing.assert_equal(pos_tails, neg_tails)

        exist = neg_edges.edata['false_neg']
        if exclude_positive:
            assert np.sum(F.asnumpy(exist) == 0) == len(exist)
        else:
            assert F.array_equal(g.has_edges_between(neg_src, neg_dst), exist)
        total_samples += batch_size
    assert total_samples <= num_edges

    # Test the knowledge graph with edge weight provied.
    total_samples = 0
    for pos_edges, neg_edges in EdgeSampler(g, batch_size,
                                            reset=False,
                                            edge_weight=edge_weight,
                                            negative_mode=mode,
                                            neg_sample_size=neg_size,
                                            exclude_positive=exclude_positive,
                                            relations=g.edata['etype'],
                                            return_false_neg=True):
        neg_lsrc, neg_ldst, neg_leid = neg_edges.all_edges(form='all', order='eid')
        neg_src = F.gather_row(neg_edges.parent_nid, neg_lsrc)
        neg_dst = F.gather_row(neg_edges.parent_nid, neg_ldst)
        neg_eid = F.gather_row(neg_edges.parent_eid, neg_leid)
        exists = neg_edges.edata['false_neg']
        neg_edges.edata['etype'] = F.gather_row(g.edata['etype'], neg_eid)
        for i in range(len(neg_eid)):
            u, v = F.asnumpy(neg_src[i]), F.asnumpy(neg_dst[i])
            if g.has_edge_between(u, v):
                eid = g.edge_ids(u, v)
                etype = g.edata['etype'][eid]
                exist = neg_edges.edata['etype'][i] == etype
                assert F.asnumpy(exists[i]) == F.asnumpy(exist)
        total_samples += batch_size
    assert total_samples <= num_edges

    # Test the knowledge graph with edge/node weight provied.
    total_samples = 0
    for pos_edges, neg_edges in EdgeSampler(g, batch_size,
                                            reset=False,
                                            edge_weight=edge_weight,
                                            node_weight=node_weight,
                                            negative_mode=mode,
                                            neg_sample_size=neg_size,
                                            exclude_positive=exclude_positive,
                                            relations=g.edata['etype'],
                                            return_false_neg=True):
        neg_lsrc, neg_ldst, neg_leid = neg_edges.all_edges(form='all', order='eid')
        neg_src = F.gather_row(neg_edges.parent_nid, neg_lsrc)
        neg_dst = F.gather_row(neg_edges.parent_nid, neg_ldst)
        neg_eid = F.gather_row(neg_edges.parent_eid, neg_leid)
        exists = neg_edges.edata['false_neg']
        neg_edges.edata['etype'] = F.gather_row(g.edata['etype'], neg_eid)
        for i in range(len(neg_eid)):
            u, v = F.asnumpy(neg_src[i]), F.asnumpy(neg_dst[i])
            if g.has_edge_between(u, v):
                eid = g.edge_ids(u, v)
                etype = g.edata['etype'][eid]
                exist = neg_edges.edata['etype'][i] == etype
                assert F.asnumpy(exists[i]) == F.asnumpy(exist)
        total_samples += batch_size
    assert total_samples <= num_edges

    # check replacement = True with pos edges no-uniform sample
    # with reset = False
    total_samples = 0
    for pos_edges, neg_edges in EdgeSampler(g, batch_size,
                                            replacement=True,
                                            reset=False,
                                            edge_weight=edge_weight,
                                            negative_mode=mode,
                                            neg_sample_size=neg_size,
                                            exclude_positive=exclude_positive,
                                            return_false_neg=True):
        _, _, pos_leid = pos_edges.all_edges(form='all', order='eid')
        assert len(pos_leid) == batch_size
        total_samples += len(pos_leid)
    assert total_samples == num_edges

    # check replacement = True with pos edges no-uniform sample
    # with reset = True
    total_samples = 0
    max_samples = 4 * num_edges
    for pos_edges, neg_edges in EdgeSampler(g, batch_size,
                                            replacement=True,
                                            reset=True,
                                            edge_weight=edge_weight,
                                            negative_mode=mode,
                                            neg_sample_size=neg_size,
                                            exclude_positive=exclude_positive,
                                            return_false_neg=True):
        _, _, pos_leid = pos_edges.all_edges(form='all', order='eid')
        assert len(pos_leid) == batch_size
        total_samples += len(pos_leid)
        if total_samples >= max_samples:
            break
    assert total_samples == max_samples

    # check replacement = False with pos/neg edges no-uniform sample
    # reset = False
    total_samples = 0
    for pos_edges, neg_edges in EdgeSampler(g, batch_size,
                                            replacement=False,
                                            reset=False,
                                            edge_weight=edge_weight,
                                            node_weight=node_weight,
                                            negative_mode=mode,
                                            neg_sample_size=neg_size,
                                            exclude_positive=exclude_positive,
                                            relations=g.edata['etype'],
                                            return_false_neg=True):
        _, _, pos_leid = pos_edges.all_edges(form='all', order='eid')
        assert len(pos_leid) == batch_size
        total_samples += len(pos_leid)
    assert total_samples == num_edges

    # check replacement = False with pos/neg edges no-uniform sample
    # reset = True
    total_samples = 0
    for pos_edges, neg_edges in EdgeSampler(g, batch_size,
                                            replacement=False,
                                            reset=True,
                                            edge_weight=edge_weight,
                                            node_weight=node_weight,
                                            negative_mode=mode,
                                            neg_sample_size=neg_size,
                                            exclude_positive=exclude_positive,
                                            relations=g.edata['etype'],
                                            return_false_neg=True):
        _, _, pos_leid = pos_edges.all_edges(form='all', order='eid')
        assert len(pos_leid) == batch_size
        total_samples += len(pos_leid)
        if total_samples >= max_samples:
            break
    assert total_samples == max_samples

    # Check Rate
    dgl.random.seed(0)
    g = generate_rand_graph(1000)
    num_edges = g.number_of_edges()
    num_nodes = g.number_of_nodes()
    edge_weight = F.copy_to(F.tensor(np.full((num_edges,), 1, dtype=np.float32)), F.cpu())
    edge_weight[0] = F.sum(edge_weight, dim=0)
    node_weight = F.copy_to(F.tensor(np.full((num_nodes,), 1, dtype=np.float32)), F.cpu())
    node_weight[-1] = F.sum(node_weight, dim=0) / 200
    etype = np.random.randint(0, 20, size=num_edges, dtype=np.int64)
    g.edata['etype'] = F.copy_to(F.tensor(etype), F.cpu())

    # Test w/o node weight.
    max_samples = num_edges // 5
    total_samples = 0
    # Test the knowledge graph with edge weight provied.
    edge_sampled = np.full((num_edges,), 0, dtype=np.int32)
    node_sampled = np.full((num_nodes,), 0, dtype=np.int32)
    for pos_edges, neg_edges in EdgeSampler(g, batch_size,
                                            replacement=True,
                                            edge_weight=edge_weight,
                                            shuffle=True,
                                            negative_mode=mode,
                                            neg_sample_size=neg_size,
                                            exclude_positive=False,
                                            relations=g.edata['etype'],
                                            return_false_neg=True):
        _, _, pos_leid = pos_edges.all_edges(form='all', order='eid')
        neg_lsrc, neg_ldst, _ = neg_edges.all_edges(form='all', order='eid')
        if 'head' in mode:
            neg_src = neg_edges.parent_nid[neg_lsrc]
            np.add.at(node_sampled, F.asnumpy(neg_src), 1)
        else:
            neg_dst = neg_edges.parent_nid[neg_ldst]
            np.add.at(node_sampled, F.asnumpy(neg_dst), 1)
        np.add.at(edge_sampled, F.asnumpy(pos_edges.parent_eid[pos_leid]), 1)
        total_samples += batch_size
        if total_samples > max_samples:
            break

    # Check rate here
    edge_rate_0 = edge_sampled[0] / edge_sampled.sum()
    edge_tail_half_cnt = edge_sampled[edge_sampled.shape[0] // 2:-1].sum()
    edge_rate_tail_half = edge_tail_half_cnt / edge_sampled.sum()
    assert np.allclose(edge_rate_0, 0.5, atol=0.05)
    assert np.allclose(edge_rate_tail_half, 0.25, atol=0.05)

    node_rate_0 = node_sampled[0] / node_sampled.sum()
    node_tail_half_cnt = node_sampled[node_sampled.shape[0] // 2:-1].sum()
    node_rate_tail_half = node_tail_half_cnt / node_sampled.sum()
    assert node_rate_0 < 0.02
    assert np.allclose(node_rate_tail_half, 0.5, atol=0.02)

    # Test the knowledge graph with edge/node weight provied.
    edge_sampled = np.full((num_edges,), 0, dtype=np.int32)
    node_sampled = np.full((num_nodes,), 0, dtype=np.int32)
    total_samples = 0
    for pos_edges, neg_edges in EdgeSampler(g, batch_size,
                                            replacement=True,
                                            edge_weight=edge_weight,
                                            node_weight=node_weight,
                                            shuffle=True,
                                            negative_mode=mode,
                                            neg_sample_size=neg_size,
                                            exclude_positive=False,
                                            relations=g.edata['etype'],
                                            return_false_neg=True):
        _, _, pos_leid = pos_edges.all_edges(form='all', order='eid')
        neg_lsrc, neg_ldst, _ = neg_edges.all_edges(form='all', order='eid')
        if 'head' in mode:
            neg_src = F.gather_row(neg_edges.parent_nid, neg_lsrc)
            np.add.at(node_sampled, F.asnumpy(neg_src), 1)
        else:
            neg_dst = F.gather_row(neg_edges.parent_nid, neg_ldst)
            np.add.at(node_sampled, F.asnumpy(neg_dst), 1)
        np.add.at(edge_sampled, F.asnumpy(pos_edges.parent_eid[pos_leid]), 1)
        total_samples += batch_size
        if total_samples > max_samples:
            break

    # Check rate here
    edge_rate_0 = edge_sampled[0] / edge_sampled.sum()
    edge_tail_half_cnt = edge_sampled[edge_sampled.shape[0] // 2:-1].sum()
    edge_rate_tail_half = edge_tail_half_cnt / edge_sampled.sum()
    assert np.allclose(edge_rate_0, 0.5, atol=0.05)
    assert np.allclose(edge_rate_tail_half, 0.25, atol=0.05)

    node_rate = node_sampled[-1] / node_sampled.sum()
    node_rate_a = np.average(node_sampled[:50]) / node_sampled.sum()
    node_rate_b = np.average(node_sampled[50:100]) / node_sampled.sum()
    # As neg sampling does not contain duplicate nodes,
    # this test takes some acceptable variation on the sample rate.
    assert np.allclose(node_rate, node_rate_a * 5, atol=0.002)
    assert np.allclose(node_rate_a, node_rate_b, atol=0.0002)

def check_positive_edge_sampler():
    g = generate_rand_graph(1000)
    num_edges = g.number_of_edges()
    edge_weight = F.copy_to(F.tensor(np.full((num_edges,), 0.1, dtype=np.float32)), F.cpu())

    edge_weight[num_edges-1] = num_edges ** 2
    EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')

    # Correctness check
    # Test the homogeneous graph.
    batch_size = 128
    edge_sampled = np.full((num_edges,), 0, dtype=np.int32)
    for pos_edges in EdgeSampler(g, batch_size,
                                    reset=False,
                                    edge_weight=edge_weight):
        _, _, pos_leid = pos_edges.all_edges(form='all', order='eid')
        np.add.at(edge_sampled, F.asnumpy(pos_edges.parent_eid[pos_leid]), 1)
    truth = np.full((num_edges,), 1, dtype=np.int32)
    edge_sampled = edge_sampled[:num_edges]
    assert np.array_equal(truth, edge_sampled)

    edge_sampled = np.full((num_edges,), 0, dtype=np.int32)
    for pos_edges in EdgeSampler(g, batch_size,
                                    reset=False,
                                    shuffle=True,
                                    edge_weight=edge_weight):
        _, _, pos_leid = pos_edges.all_edges(form='all', order='eid')
        np.add.at(edge_sampled, F.asnumpy(pos_edges.parent_eid[pos_leid]), 1)
    truth = np.full((num_edges,), 1, dtype=np.int32)
    edge_sampled = edge_sampled[:num_edges]
    assert np.array_equal(truth, edge_sampled)


@unittest.skipIf(dgl.backend.backend_name == "tensorflow", reason="TF doesn't support item assignment")
def test_negative_sampler():
    check_negative_sampler('chunk-head', False, 10)
    check_negative_sampler('head', True, 10)
    check_negative_sampler('head', False, 10)
    check_weighted_negative_sampler('chunk-head', False, 10)
    check_weighted_negative_sampler('head', True, 10)
    check_weighted_negative_sampler('head', False, 10)
    check_positive_edge_sampler()
    #disable this check for now. It might take too long time.
    #check_negative_sampler('head', False, 100)


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
