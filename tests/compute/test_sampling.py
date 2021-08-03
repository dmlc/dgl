import dgl
import backend as F
import numpy as np
import unittest
from collections import defaultdict

def check_random_walk(g, metapath, traces, ntypes, prob=None, trace_eids=None):
    traces = F.asnumpy(traces)
    ntypes = F.asnumpy(ntypes)
    for j in range(traces.shape[1] - 1):
        assert ntypes[j] == g.get_ntype_id(g.to_canonical_etype(metapath[j])[0])
        assert ntypes[j + 1] == g.get_ntype_id(g.to_canonical_etype(metapath[j])[2])

    for i in range(traces.shape[0]):
        for j in range(traces.shape[1] - 1):
            assert g.has_edge_between(
                traces[i, j], traces[i, j+1], etype=metapath[j])
            if prob is not None and prob in g.edges[metapath[j]].data:
                p = F.asnumpy(g.edges[metapath[j]].data['p'])
                eids = g.edge_ids(traces[i, j], traces[i, j+1], etype=metapath[j])
                assert p[eids] != 0
            if trace_eids is not None:
                u, v = g.find_edges(trace_eids[i, j], etype=metapath[j])
                assert (u == traces[i, j]) and (v == traces[i, j + 1])

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU random walk not implemented")
def test_random_walk():
    g1 = dgl.heterograph({
        ('user', 'follow', 'user'): ([0, 1, 2], [1, 2, 0])
        })
    g2 = dgl.heterograph({
        ('user', 'follow', 'user'): ([0, 1, 1, 2, 3], [1, 2, 3, 0, 0])
        })
    g3 = dgl.heterograph({
        ('user', 'follow', 'user'): ([0, 1, 2], [1, 2, 0]),
        ('user', 'view', 'item'): ([0, 1, 2], [0, 1, 2]),
        ('item', 'viewed-by', 'user'): ([0, 1, 2], [0, 1, 2])})
    g4 = dgl.heterograph({
        ('user', 'follow', 'user'): ([0, 1, 1, 2, 3], [1, 2, 3, 0, 0]),
        ('user', 'view', 'item'): ([0, 0, 1, 2, 3, 3], [0, 1, 1, 2, 2, 1]),
        ('item', 'viewed-by', 'user'): ([0, 1, 1, 2, 2, 1], [0, 0, 1, 2, 3, 3])})

    g2.edata['p'] = F.tensor([3, 0, 3, 3, 3], dtype=F.float32)
    g2.edata['p2'] = F.tensor([[3], [0], [3], [3], [3]], dtype=F.float32)
    g4.edges['follow'].data['p'] = F.tensor([3, 0, 3, 3, 3], dtype=F.float32)
    g4.edges['viewed-by'].data['p'] = F.tensor([1, 1, 1, 1, 1, 1], dtype=F.float32)

    traces, eids, ntypes = dgl.sampling.random_walk(g1, [0, 1, 2, 0, 1, 2], length=4, return_eids=True)
    check_random_walk(g1, ['follow'] * 4, traces, ntypes, trace_eids=eids)
    traces, eids, ntypes = dgl.sampling.random_walk(g1, [0, 1, 2, 0, 1, 2], length=4, restart_prob=0., return_eids=True)
    check_random_walk(g1, ['follow'] * 4, traces, ntypes, trace_eids=eids)
    traces, ntypes = dgl.sampling.random_walk(
        g1, [0, 1, 2, 0, 1, 2], length=4, restart_prob=F.zeros((4,), F.float32, F.cpu()))
    check_random_walk(g1, ['follow'] * 4, traces, ntypes)
    traces, ntypes = dgl.sampling.random_walk(
        g1, [0, 1, 2, 0, 1, 2], length=5,
        restart_prob=F.tensor([0, 0, 0, 0, 1], dtype=F.float32))
    check_random_walk(
        g1, ['follow'] * 4, F.slice_axis(traces, 1, 0, 5), F.slice_axis(ntypes, 0, 0, 5))
    assert (F.asnumpy(traces)[:, 5] == -1).all()

    traces, eids, ntypes = dgl.sampling.random_walk(
        g2, [0, 1, 2, 3, 0, 1, 2, 3], length=4, return_eids=True)
    check_random_walk(g2, ['follow'] * 4, traces, ntypes, trace_eids=eids)

    traces, eids, ntypes = dgl.sampling.random_walk(
        g2, [0, 1, 2, 3, 0, 1, 2, 3], length=4, prob='p', return_eids=True)
    check_random_walk(g2, ['follow'] * 4, traces, ntypes, 'p', trace_eids=eids)

    try:
        traces, ntypes = dgl.sampling.random_walk(
            g2, [0, 1, 2, 3, 0, 1, 2, 3], length=4, prob='p2')
        fail = False
    except dgl.DGLError:
        fail = True
    assert fail

    metapath = ['follow', 'view', 'viewed-by'] * 2
    traces, eids, ntypes = dgl.sampling.random_walk(
        g3, [0, 1, 2, 0, 1, 2], metapath=metapath, return_eids=True)
    check_random_walk(g3, metapath, traces, ntypes, trace_eids=eids)

    metapath = ['follow', 'view', 'viewed-by'] * 2
    traces, eids, ntypes = dgl.sampling.random_walk(
        g4, [0, 1, 2, 3, 0, 1, 2, 3], metapath=metapath, return_eids=True)
    check_random_walk(g4, metapath, traces, ntypes, trace_eids=eids)

    traces, eids, ntypes = dgl.sampling.random_walk(
        g4, [0, 1, 2, 0, 1, 2], metapath=metapath, return_eids=True)
    check_random_walk(g4, metapath, traces, ntypes, trace_eids=eids)

    metapath = ['follow', 'view', 'viewed-by'] * 2
    traces, eids, ntypes = dgl.sampling.random_walk(
        g4, [0, 1, 2, 3, 0, 1, 2, 3], metapath=metapath, prob='p', return_eids=True)
    check_random_walk(g4, metapath, traces, ntypes, 'p', trace_eids=eids)
    traces, eids, ntypes = dgl.sampling.random_walk(
        g4, [0, 1, 2, 3, 0, 1, 2, 3], metapath=metapath, prob='p', restart_prob=0., return_eids=True)
    check_random_walk(g4, metapath, traces, ntypes, 'p', trace_eids=eids)
    traces, eids, ntypes = dgl.sampling.random_walk(
        g4, [0, 1, 2, 3, 0, 1, 2, 3], metapath=metapath, prob='p',
        restart_prob=F.zeros((6,), F.float32, F.cpu()), return_eids=True)
    check_random_walk(g4, metapath, traces, ntypes, 'p', trace_eids=eids)
    traces, eids, ntypes = dgl.sampling.random_walk(
        g4, [0, 1, 2, 3, 0, 1, 2, 3], metapath=metapath + ['follow'], prob='p',
        restart_prob=F.tensor([0, 0, 0, 0, 0, 0, 1], F.float32), return_eids=True)
    check_random_walk(g4, metapath, traces[:, :7], ntypes[:7], 'p', trace_eids=eids)
    assert (F.asnumpy(traces[:, 7]) == -1).all()

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU random walk not implemented")
def test_node2vec():
    g1 = dgl.heterograph({
        ('user', 'follow', 'user'): ([0, 1, 2], [1, 2, 0])
        })
    g2 = dgl.heterograph({
        ('user', 'follow', 'user'): ([0, 1, 1, 2, 3], [1, 2, 3, 0, 0])
        })
    g2.edata['p'] = F.tensor([3, 0, 3, 3, 3], dtype=F.float32)

    ntypes = F.zeros((5,), dtype=F.int64)

    traces, eids = dgl.sampling.node2vec_random_walk(g1, [0, 1, 2, 0, 1, 2], 1, 1, 4, return_eids=True)
    check_random_walk(g1, ['follow'] * 4, traces, ntypes, trace_eids=eids)

    traces, eids = dgl.sampling.node2vec_random_walk(
        g2, [0, 1, 2, 3, 0, 1, 2, 3], 1, 1, 4, prob='p', return_eids=True)
    check_random_walk(g2, ['follow'] * 4, traces, ntypes, 'p', trace_eids=eids)

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU pack traces not implemented")
def test_pack_traces():
    traces, types = (np.array(
        [[ 0,  1, -1, -1, -1, -1, -1],
         [ 0,  1,  1,  3,  0,  0,  0]], dtype='int64'),
        np.array([0, 0, 1, 0, 0, 1, 0], dtype='int64'))
    traces = F.zerocopy_from_numpy(traces)
    types = F.zerocopy_from_numpy(types)
    result = dgl.sampling.pack_traces(traces, types)
    assert F.array_equal(result[0], F.tensor([0, 1, 0, 1, 1, 3, 0, 0, 0], dtype=F.int64))
    assert F.array_equal(result[1], F.tensor([0, 0, 0, 0, 1, 0, 0, 1, 0], dtype=F.int64))
    assert F.array_equal(result[2], F.tensor([2, 7], dtype=F.int64))
    assert F.array_equal(result[3], F.tensor([0, 2], dtype=F.int64))

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU not implemented")
def test_pinsage_sampling():
    def _test_sampler(g, sampler, ntype):
        neighbor_g = sampler(F.tensor([0, 2], dtype=F.int64))
        assert neighbor_g.ntypes == [ntype]
        u, v = neighbor_g.all_edges(form='uv', order='eid')
        uv = list(zip(F.asnumpy(u).tolist(), F.asnumpy(v).tolist()))
        assert (1, 0) in uv or (0, 0) in uv
        assert (2, 2) in uv or (3, 2) in uv

    g = dgl.heterograph({
        ('item', 'bought-by', 'user'): ([0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 2, 3, 2, 3]),
        ('user', 'bought', 'item'): ([0, 1, 0, 1, 2, 3, 2, 3], [0, 0, 1, 1, 2, 2, 3, 3])})
    sampler = dgl.sampling.PinSAGESampler(g, 'item', 'user', 4, 0.5, 3, 2)
    _test_sampler(g, sampler, 'item')
    sampler = dgl.sampling.RandomWalkNeighborSampler(g, 4, 0.5, 3, 2, ['bought-by', 'bought'])
    _test_sampler(g, sampler, 'item')
    sampler = dgl.sampling.RandomWalkNeighborSampler(g, 4, 0.5, 3, 2,
        [('item', 'bought-by', 'user'), ('user', 'bought', 'item')])
    _test_sampler(g, sampler, 'item')
    g = dgl.graph(([0, 0, 1, 1, 2, 2, 3, 3],
                   [0, 1, 0, 1, 2, 3, 2, 3]))
    sampler = dgl.sampling.RandomWalkNeighborSampler(g, 4, 0.5, 3, 2)
    _test_sampler(g, sampler, g.ntypes[0])
    g = dgl.heterograph({
        ('A', 'AB', 'B'): ([0, 2], [1, 3]),
        ('B', 'BC', 'C'): ([1, 3], [2, 1]),
        ('C', 'CA', 'A'): ([2, 1], [0, 2])})
    sampler = dgl.sampling.RandomWalkNeighborSampler(g, 4, 0.5, 3, 2, ['AB', 'BC', 'CA'])
    _test_sampler(g, sampler, 'A')

def _gen_neighbor_sampling_test_graph(hypersparse, reverse):
    if hypersparse:
        # should crash if allocated a CSR
        card = 1 << 50
        num_nodes_dict = {'user': card, 'game': card, 'coin': card}
    else:
        card = None
        num_nodes_dict = None

    if reverse:
        g = dgl.heterograph({
            ('user', 'follow', 'user'): ([0, 0, 0, 1, 1, 1, 2], [1, 2, 3, 0, 2, 3, 0])
        }, {'user': card if card is not None else 4})
        g = g.to(F.ctx())
        g.edata['prob'] = F.tensor([.5, .5, 0., .5, .5, 0., 1.], dtype=F.float32)
        hg = dgl.heterograph({
            ('user', 'follow', 'user'): ([0, 0, 0, 1, 1, 1, 2],
                                         [1, 2, 3, 0, 2, 3, 0]),
            ('game', 'play', 'user'): ([0, 1, 2, 2], [0, 0, 1, 3]),
            ('user', 'liked-by', 'game'): ([0, 1, 2, 0, 3, 0], [2, 2, 2, 1, 1, 0]),
            ('coin', 'flips', 'user'): ([0, 0, 0, 0], [0, 1, 2, 3])
        }, num_nodes_dict)
        hg = hg.to(F.ctx())
    else:
        g = dgl.heterograph({
            ('user', 'follow', 'user'): ([1, 2, 3, 0, 2, 3, 0], [0, 0, 0, 1, 1, 1, 2])
        }, {'user': card if card is not None else 4})
        g = g.to(F.ctx())
        g.edata['prob'] = F.tensor([.5, .5, 0., .5, .5, 0., 1.], dtype=F.float32)
        hg = dgl.heterograph({
            ('user', 'follow', 'user'): ([1, 2, 3, 0, 2, 3, 0],
                                         [0, 0, 0, 1, 1, 1, 2]),
            ('user', 'play', 'game'): ([0, 0, 1, 3], [0, 1, 2, 2]),
            ('game', 'liked-by', 'user'): ([2, 2, 2, 1, 1, 0], [0, 1, 2, 0, 3, 0]),
            ('user', 'flips', 'coin'): ([0, 1, 2, 3], [0, 0, 0, 0])
        }, num_nodes_dict)
        hg = hg.to(F.ctx())
    hg.edges['follow'].data['prob'] = F.tensor([.5, .5, 0., .5, .5, 0., 1.], dtype=F.float32)
    hg.edges['play'].data['prob'] = F.tensor([.8, .5, .5, .5], dtype=F.float32)
    hg.edges['liked-by'].data['prob'] = F.tensor([.3, .5, .2, .5, .1, .1], dtype=F.float32)

    return g, hg

def _gen_neighbor_topk_test_graph(hypersparse, reverse):
    if hypersparse:
        # should crash if allocated a CSR
        card = 1 << 50
    else:
        card = None

    if reverse:
        g = dgl.heterograph({
            ('user', 'follow', 'user'): ([0, 0, 0, 1, 1, 1, 2], [1, 2, 3, 0, 2, 3, 0])
        })
        g.edata['weight'] = F.tensor([.5, .3, 0., -5., 22., 0., 1.], dtype=F.float32)
        hg = dgl.heterograph({
            ('user', 'follow', 'user'): ([0, 0, 0, 1, 1, 1, 2],
                                         [1, 2, 3, 0, 2, 3, 0]),
            ('game', 'play', 'user'): ([0, 1, 2, 2], [0, 0, 1, 3]),
            ('user', 'liked-by', 'game'): ([0, 1, 2, 0, 3, 0], [2, 2, 2, 1, 1, 0]),
            ('coin', 'flips', 'user'): ([0, 0, 0, 0], [0, 1, 2, 3])
        })
    else:
        g = dgl.heterograph({
            ('user', 'follow', 'user'): ([1, 2, 3, 0, 2, 3, 0], [0, 0, 0, 1, 1, 1, 2])
        })
        g.edata['weight'] = F.tensor([.5, .3, 0., -5., 22., 0., 1.], dtype=F.float32)
        hg = dgl.heterograph({
            ('user', 'follow', 'user'): ([1, 2, 3, 0, 2, 3, 0],
                                         [0, 0, 0, 1, 1, 1, 2]),
            ('user', 'play', 'game'): ([0, 0, 1, 3], [0, 1, 2, 2]),
            ('game', 'liked-by', 'user'): ([2, 2, 2, 1, 1, 0], [0, 1, 2, 0, 3, 0]),
            ('user', 'flips', 'coin'): ([0, 1, 2, 3], [0, 0, 0, 0])
        })
    hg.edges['follow'].data['weight'] = F.tensor([.5, .3, 0., -5., 22., 0., 1.], dtype=F.float32)
    hg.edges['play'].data['weight'] = F.tensor([.8, .5, .4, .5], dtype=F.float32)
    hg.edges['liked-by'].data['weight'] = F.tensor([.3, .5, .2, .5, .1, .1], dtype=F.float32)
    hg.edges['flips'].data['weight'] = F.tensor([10, 2, 13, -1], dtype=F.float32)
    return g, hg

def _test_sample_neighbors(hypersparse, prob):
    g, hg = _gen_neighbor_sampling_test_graph(hypersparse, False)

    def _test1(p, replace):
        subg = dgl.sampling.sample_neighbors(g, [0, 1], -1, prob=p, replace=replace)
        assert subg.number_of_nodes() == g.number_of_nodes()
        u, v = subg.edges()
        u_ans, v_ans = subg.in_edges([0, 1])
        uv = set(zip(F.asnumpy(u), F.asnumpy(v)))
        uv_ans = set(zip(F.asnumpy(u_ans), F.asnumpy(v_ans)))
        assert uv == uv_ans

        for i in range(10):
            subg = dgl.sampling.sample_neighbors(g, [0, 1], 2, prob=p, replace=replace)
            assert subg.number_of_nodes() == g.number_of_nodes()
            assert subg.number_of_edges() == 4
            u, v = subg.edges()
            assert set(F.asnumpy(F.unique(v))) == {0, 1}
            assert F.array_equal(F.astype(g.has_edges_between(u, v), F.int64), F.ones((4,), dtype=F.int64))
            assert F.array_equal(g.edge_ids(u, v), subg.edata[dgl.EID])
            edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
            if not replace:
                # check no duplication
                assert len(edge_set) == 4
            if p is not None:
                assert not (3, 0) in edge_set
                assert not (3, 1) in edge_set
    _test1(prob, True)   # w/ replacement, uniform
    _test1(prob, False)  # w/o replacement, uniform

    def _test2(p, replace):  # fanout > #neighbors
        subg = dgl.sampling.sample_neighbors(g, [0, 2], -1, prob=p, replace=replace)
        assert subg.number_of_nodes() == g.number_of_nodes()
        u, v = subg.edges()
        u_ans, v_ans = subg.in_edges([0, 2])
        uv = set(zip(F.asnumpy(u), F.asnumpy(v)))
        uv_ans = set(zip(F.asnumpy(u_ans), F.asnumpy(v_ans)))
        assert uv == uv_ans

        for i in range(10):
            subg = dgl.sampling.sample_neighbors(g, [0, 2], 2, prob=p, replace=replace)
            assert subg.number_of_nodes() == g.number_of_nodes()
            num_edges = 4 if replace else 3
            assert subg.number_of_edges() == num_edges
            u, v = subg.edges()
            assert set(F.asnumpy(F.unique(v))) == {0, 2}
            assert F.array_equal(F.astype(g.has_edges_between(u, v), F.int64), F.ones((num_edges,), dtype=F.int64))
            assert F.array_equal(g.edge_ids(u, v), subg.edata[dgl.EID])
            edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
            if not replace:
                # check no duplication
                assert len(edge_set) == num_edges
            if p is not None:
                assert not (3, 0) in edge_set
    _test2(prob, True)   # w/ replacement, uniform
    _test2(prob, False)  # w/o replacement, uniform

    def _test3(p, replace):
        subg = dgl.sampling.sample_neighbors(hg, {'user': [0, 1], 'game': 0}, -1, prob=p, replace=replace)
        assert len(subg.ntypes) == 3
        assert len(subg.etypes) == 4
        assert subg['follow'].number_of_edges() == 6
        assert subg['play'].number_of_edges() == 1
        assert subg['liked-by'].number_of_edges() == 4
        assert subg['flips'].number_of_edges() == 0

        for i in range(10):
            subg = dgl.sampling.sample_neighbors(hg, {'user' : [0,1], 'game' : 0}, 2, prob=p, replace=replace)
            assert len(subg.ntypes) == 3
            assert len(subg.etypes) == 4
            assert subg['follow'].number_of_edges() == 4
            assert subg['play'].number_of_edges() == 2 if replace else 1
            assert subg['liked-by'].number_of_edges() == 4 if replace else 3
            assert subg['flips'].number_of_edges() == 0

    _test3(prob, True)   # w/ replacement, uniform
    _test3(prob, False)  # w/o replacement, uniform

    # test different fanouts for different relations
    for i in range(10):
        subg = dgl.sampling.sample_neighbors(
            hg,
            {'user' : [0,1], 'game' : 0, 'coin': 0},
            {'follow': 1, 'play': 2, 'liked-by': 0, 'flips': -1},
            replace=True)
        assert len(subg.ntypes) == 3
        assert len(subg.etypes) == 4
        assert subg['follow'].number_of_edges() == 2
        assert subg['play'].number_of_edges() == 2
        assert subg['liked-by'].number_of_edges() == 0
        assert subg['flips'].number_of_edges() == 4

def _test_sample_neighbors_outedge(hypersparse):
    g, hg = _gen_neighbor_sampling_test_graph(hypersparse, True)

    def _test1(p, replace):
        subg = dgl.sampling.sample_neighbors(g, [0, 1], -1, prob=p, replace=replace, edge_dir='out')
        assert subg.number_of_nodes() == g.number_of_nodes()
        u, v = subg.edges()
        u_ans, v_ans = subg.out_edges([0, 1])
        uv = set(zip(F.asnumpy(u), F.asnumpy(v)))
        uv_ans = set(zip(F.asnumpy(u_ans), F.asnumpy(v_ans)))
        assert uv == uv_ans

        for i in range(10):
            subg = dgl.sampling.sample_neighbors(g, [0, 1], 2, prob=p, replace=replace, edge_dir='out')
            assert subg.number_of_nodes() == g.number_of_nodes()
            assert subg.number_of_edges() == 4
            u, v = subg.edges()
            assert set(F.asnumpy(F.unique(u))) == {0, 1}
            assert F.array_equal(F.astype(g.has_edges_between(u, v), F.int64), F.ones((4,), dtype=F.int64))
            assert F.array_equal(g.edge_ids(u, v), subg.edata[dgl.EID])
            edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
            if not replace:
                # check no duplication
                assert len(edge_set) == 4
            if p is not None:
                assert not (0, 3) in edge_set
                assert not (1, 3) in edge_set
    _test1(None, True)   # w/ replacement, uniform
    _test1(None, False)  # w/o replacement, uniform
    _test1('prob', True)   # w/ replacement
    _test1('prob', False)  # w/o replacement

    def _test2(p, replace):  # fanout > #neighbors
        subg = dgl.sampling.sample_neighbors(g, [0, 2], -1, prob=p, replace=replace, edge_dir='out')
        assert subg.number_of_nodes() == g.number_of_nodes()
        u, v = subg.edges()
        u_ans, v_ans = subg.out_edges([0, 2])
        uv = set(zip(F.asnumpy(u), F.asnumpy(v)))
        uv_ans = set(zip(F.asnumpy(u_ans), F.asnumpy(v_ans)))
        assert uv == uv_ans

        for i in range(10):
            subg = dgl.sampling.sample_neighbors(g, [0, 2], 2, prob=p, replace=replace, edge_dir='out')
            assert subg.number_of_nodes() == g.number_of_nodes()
            num_edges = 4 if replace else 3
            assert subg.number_of_edges() == num_edges
            u, v = subg.edges()
            assert set(F.asnumpy(F.unique(u))) == {0, 2}
            assert F.array_equal(F.astype(g.has_edges_between(u, v), F.int64), F.ones((num_edges,), dtype=F.int64))
            assert F.array_equal(g.edge_ids(u, v), subg.edata[dgl.EID])
            edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
            if not replace:
                # check no duplication
                assert len(edge_set) == num_edges
            if p is not None:
                assert not (0, 3) in edge_set
    _test2(None, True)   # w/ replacement, uniform
    _test2(None, False)  # w/o replacement, uniform
    _test2('prob', True)   # w/ replacement
    _test2('prob', False)  # w/o replacement

    def _test3(p, replace):
        subg = dgl.sampling.sample_neighbors(hg, {'user': [0, 1], 'game': 0}, -1, prob=p, replace=replace, edge_dir='out')
        assert len(subg.ntypes) == 3
        assert len(subg.etypes) == 4
        assert subg['follow'].number_of_edges() == 6
        assert subg['play'].number_of_edges() == 1
        assert subg['liked-by'].number_of_edges() == 4
        assert subg['flips'].number_of_edges() == 0

        for i in range(10):
            subg = dgl.sampling.sample_neighbors(hg, {'user' : [0,1], 'game' : 0}, 2, prob=p, replace=replace, edge_dir='out')
            assert len(subg.ntypes) == 3
            assert len(subg.etypes) == 4
            assert subg['follow'].number_of_edges() == 4
            assert subg['play'].number_of_edges() == 2 if replace else 1
            assert subg['liked-by'].number_of_edges() == 4 if replace else 3
            assert subg['flips'].number_of_edges() == 0

    _test3(None, True)   # w/ replacement, uniform
    _test3(None, False)  # w/o replacement, uniform
    _test3('prob', True)   # w/ replacement
    _test3('prob', False)  # w/o replacement

def _test_sample_neighbors_topk(hypersparse):
    g, hg = _gen_neighbor_topk_test_graph(hypersparse, False)

    def _test1():
        subg = dgl.sampling.select_topk(g, -1, 'weight', [0, 1])
        assert subg.number_of_nodes() == g.number_of_nodes()
        u, v = subg.edges()
        u_ans, v_ans = subg.in_edges([0, 1])
        uv = set(zip(F.asnumpy(u), F.asnumpy(v)))
        uv_ans = set(zip(F.asnumpy(u_ans), F.asnumpy(v_ans)))
        assert uv == uv_ans

        subg = dgl.sampling.select_topk(g, 2, 'weight', [0, 1])
        assert subg.number_of_nodes() == g.number_of_nodes()
        assert subg.number_of_edges() == 4
        u, v = subg.edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert F.array_equal(g.edge_ids(u, v), subg.edata[dgl.EID])
        assert edge_set == {(2,0),(1,0),(2,1),(3,1)}
    _test1()

    def _test2():  # k > #neighbors
        subg = dgl.sampling.select_topk(g, -1, 'weight', [0, 2])
        assert subg.number_of_nodes() == g.number_of_nodes()
        u, v = subg.edges()
        u_ans, v_ans = subg.in_edges([0, 2])
        uv = set(zip(F.asnumpy(u), F.asnumpy(v)))
        uv_ans = set(zip(F.asnumpy(u_ans), F.asnumpy(v_ans)))
        assert uv == uv_ans

        subg = dgl.sampling.select_topk(g, 2, 'weight', [0, 2])
        assert subg.number_of_nodes() == g.number_of_nodes()
        assert subg.number_of_edges() == 3
        u, v = subg.edges()
        assert F.array_equal(g.edge_ids(u, v), subg.edata[dgl.EID])
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert edge_set == {(2,0),(1,0),(0,2)}
    _test2()

    def _test3():
        subg = dgl.sampling.select_topk(hg, 2, 'weight', {'user' : [0,1], 'game' : 0})
        assert len(subg.ntypes) == 3
        assert len(subg.etypes) == 4
        u, v = subg['follow'].edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert F.array_equal(hg['follow'].edge_ids(u, v), subg['follow'].edata[dgl.EID])
        assert edge_set == {(2,0),(1,0),(2,1),(3,1)}
        u, v = subg['play'].edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert F.array_equal(hg['play'].edge_ids(u, v), subg['play'].edata[dgl.EID])
        assert edge_set == {(0,0)}
        u, v = subg['liked-by'].edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert F.array_equal(hg['liked-by'].edge_ids(u, v), subg['liked-by'].edata[dgl.EID])
        assert edge_set == {(2,0),(2,1),(1,0)}
        assert subg['flips'].number_of_edges() == 0
    _test3()

    # test different k for different relations
    subg = dgl.sampling.select_topk(
        hg, {'follow': 1, 'play': 2, 'liked-by': 0, 'flips': -1}, 'weight', {'user' : [0,1], 'game' : 0, 'coin': 0})
    assert len(subg.ntypes) == 3
    assert len(subg.etypes) == 4
    assert subg['follow'].number_of_edges() == 2
    assert subg['play'].number_of_edges() == 1
    assert subg['liked-by'].number_of_edges() == 0
    assert subg['flips'].number_of_edges() == 4

def _test_sample_neighbors_topk_outedge(hypersparse):
    g, hg = _gen_neighbor_topk_test_graph(hypersparse, True)

    def _test1():
        subg = dgl.sampling.select_topk(g, -1, 'weight', [0, 1], edge_dir='out')
        assert subg.number_of_nodes() == g.number_of_nodes()
        u, v = subg.edges()
        u_ans, v_ans = subg.out_edges([0, 1])
        uv = set(zip(F.asnumpy(u), F.asnumpy(v)))
        uv_ans = set(zip(F.asnumpy(u_ans), F.asnumpy(v_ans)))
        assert uv == uv_ans

        subg = dgl.sampling.select_topk(g, 2, 'weight', [0, 1], edge_dir='out')
        assert subg.number_of_nodes() == g.number_of_nodes()
        assert subg.number_of_edges() == 4
        u, v = subg.edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert F.array_equal(g.edge_ids(u, v), subg.edata[dgl.EID])
        assert edge_set == {(0,2),(0,1),(1,2),(1,3)}
    _test1()

    def _test2():  # k > #neighbors
        subg = dgl.sampling.select_topk(g, -1, 'weight', [0, 2], edge_dir='out')
        assert subg.number_of_nodes() == g.number_of_nodes()
        u, v = subg.edges()
        u_ans, v_ans = subg.out_edges([0, 2])
        uv = set(zip(F.asnumpy(u), F.asnumpy(v)))
        uv_ans = set(zip(F.asnumpy(u_ans), F.asnumpy(v_ans)))
        assert uv == uv_ans

        subg = dgl.sampling.select_topk(g, 2, 'weight', [0, 2], edge_dir='out')
        assert subg.number_of_nodes() == g.number_of_nodes()
        assert subg.number_of_edges() == 3
        u, v = subg.edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert F.array_equal(g.edge_ids(u, v), subg.edata[dgl.EID])
        assert edge_set == {(0,2),(0,1),(2,0)}
    _test2()

    def _test3():
        subg = dgl.sampling.select_topk(hg, 2, 'weight', {'user' : [0,1], 'game' : 0}, edge_dir='out')
        assert len(subg.ntypes) == 3
        assert len(subg.etypes) == 4
        u, v = subg['follow'].edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert F.array_equal(hg['follow'].edge_ids(u, v), subg['follow'].edata[dgl.EID])
        assert edge_set == {(0,2),(0,1),(1,2),(1,3)}
        u, v = subg['play'].edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert F.array_equal(hg['play'].edge_ids(u, v), subg['play'].edata[dgl.EID])
        assert edge_set == {(0,0)}
        u, v = subg['liked-by'].edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert F.array_equal(hg['liked-by'].edge_ids(u, v), subg['liked-by'].edata[dgl.EID])
        assert edge_set == {(0,2),(1,2),(0,1)}
        assert subg['flips'].number_of_edges() == 0
    _test3()

def test_sample_neighbors_noprob():
    _test_sample_neighbors(False, None)
    #_test_sample_neighbors(True)

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sample neighbors with probability is not implemented")
def test_sample_neighbors_prob():
    _test_sample_neighbors(False, 'prob')
    #_test_sample_neighbors(True)

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sample neighbors not implemented")
def test_sample_neighbors_outedge():
    _test_sample_neighbors_outedge(False)
    #_test_sample_neighbors_outedge(True)

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sample neighbors not implemented")
def test_sample_neighbors_topk():
    _test_sample_neighbors_topk(False)
    #_test_sample_neighbors_topk(True)

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sample neighbors not implemented")
def test_sample_neighbors_topk_outedge():
    _test_sample_neighbors_topk_outedge(False)
    #_test_sample_neighbors_topk_outedge(True)

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sample neighbors not implemented")
def test_sample_neighbors_with_0deg():
    g = dgl.graph(([], []), num_nodes=5)
    sg = dgl.sampling.sample_neighbors(g, F.tensor([1, 2], dtype=F.int64), 2, edge_dir='in', replace=False)
    assert sg.number_of_edges() == 0
    sg = dgl.sampling.sample_neighbors(g, F.tensor([1, 2], dtype=F.int64), 2, edge_dir='in', replace=True)
    assert sg.number_of_edges() == 0
    sg = dgl.sampling.sample_neighbors(g, F.tensor([1, 2], dtype=F.int64), 2, edge_dir='out', replace=False)
    assert sg.number_of_edges() == 0
    sg = dgl.sampling.sample_neighbors(g, F.tensor([1, 2], dtype=F.int64), 2, edge_dir='out', replace=True)
    assert sg.number_of_edges() == 0

def create_test_graph(num_nodes, num_edges_per_node, bipartite=False):
    src = np.concatenate(
        [np.array([i] * num_edges_per_node) for i in range(num_nodes)])
    dst = np.concatenate(
        [np.random.choice(num_nodes, num_edges_per_node, replace=False) for i in range(num_nodes)]
    )
    if bipartite:
        g = dgl.heterograph({("u", "e", "v") : (src, dst)})
    else:
        g = dgl.graph((src, dst))
    return g

def create_etype_test_graph(num_nodes, num_edges_per_node, rare_cnt):
    src = np.concatenate(
        [np.random.choice(num_nodes, num_edges_per_node, replace=False) for i in range(num_nodes)]
    )
    dst = np.concatenate(
        [np.array([i] * num_edges_per_node) for i in range(num_nodes)])

    minor_src = np.concatenate(
        [np.random.choice(num_nodes, 2, replace=False) for i in range(num_nodes)]
    )
    minor_dst = np.concatenate(
        [np.array([i] * 2) for i in range(num_nodes)])

    most_zero_src = np.concatenate(
        [np.random.choice(num_nodes, num_edges_per_node, replace=False) for i in range(rare_cnt)]
    )
    most_zero_dst = np.concatenate(
        [np.array([i] * num_edges_per_node) for i in range(rare_cnt)])


    g = dgl.heterograph({("v", "e_major", "u") : (src, dst),
                         ("u", "e_major_rev", "v") : (dst, src),
                         ("v2", "e_minor", "u") : (minor_src, minor_dst),
                         ("v2", "most_zero", "u") : (most_zero_src, most_zero_dst),
                         ("u", "e_minor_rev", "v2") : (minor_dst, minor_src)})

    return g

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sample neighbors not implemented")
def test_sample_neighbors_biased_homogeneous():
    g = create_test_graph(100, 30)

    def check_num(nodes, tag):
        nodes, tag = F.asnumpy(nodes), F.asnumpy(tag)
        cnt = [sum(tag[nodes] == i) for i in range(4)]
        # No tag 0
        assert cnt[0] == 0

        # very rare tag 1
        assert cnt[2] > 2 * cnt[1]
        assert cnt[3] > 2 * cnt[1]

    tag = F.tensor(np.random.choice(4, 100))
    bias = F.tensor([0, 0.1, 10, 10], dtype=F.float32)
    # inedge / without replacement
    g_sorted = dgl.sort_csc_by_tag(g, tag)
    for _ in range(5):
        subg = dgl.sampling.sample_neighbors_biased(g_sorted, g.nodes(), 5, bias, replace=False)
        check_num(subg.edges()[0], tag)
        u, v = subg.edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert len(edge_set) == subg.number_of_edges()

    # inedge / with replacement
    for _ in range(5):
        subg = dgl.sampling.sample_neighbors_biased(g_sorted, g.nodes(), 5, bias, replace=True)
        check_num(subg.edges()[0], tag)

    # outedge / without replacement
    g_sorted = dgl.sort_csr_by_tag(g, tag)
    for _ in range(5):
        subg = dgl.sampling.sample_neighbors_biased(g_sorted, g.nodes(), 5, bias, edge_dir='out', replace=False)
        check_num(subg.edges()[1], tag)
        u, v = subg.edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert len(edge_set) == subg.number_of_edges()

    # outedge / with replacement
    for _ in range(5):
        subg = dgl.sampling.sample_neighbors_biased(g_sorted, g.nodes(), 5, bias, edge_dir='out', replace=True)
        check_num(subg.edges()[1], tag)

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sample neighbors not implemented")
def test_sample_neighbors_biased_bipartite():
    g = create_test_graph(100, 30, True)
    num_dst = g.number_of_dst_nodes()
    bias = F.tensor([0, 0.01, 10, 10], dtype=F.float32)
    def check_num(nodes, tag):
        nodes, tag = F.asnumpy(nodes), F.asnumpy(tag)
        cnt = [sum(tag[nodes] == i) for i in range(4)]
        # No tag 0
        assert cnt[0] == 0

        # very rare tag 1
        assert cnt[2] > 2 * cnt[1]
        assert cnt[3] > 2 * cnt[1]

    # inedge / without replacement
    tag = F.tensor(np.random.choice(4, 100))
    g_sorted = dgl.sort_csc_by_tag(g, tag)
    for _ in range(5):
        subg = dgl.sampling.sample_neighbors_biased(g_sorted, g.dstnodes(), 5, bias, replace=False)
        check_num(subg.edges()[0], tag)
        u, v = subg.edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert len(edge_set) == subg.number_of_edges()

    # inedge / with replacement
    for _ in range(5):
        subg = dgl.sampling.sample_neighbors_biased(g_sorted, g.dstnodes(), 5, bias, replace=True)
        check_num(subg.edges()[0], tag)

    # outedge / without replacement
    tag = F.tensor(np.random.choice(4, num_dst))
    g_sorted = dgl.sort_csr_by_tag(g, tag)
    for _ in range(5):
        subg = dgl.sampling.sample_neighbors_biased(g_sorted, g.srcnodes(), 5, bias, edge_dir='out', replace=False)
        check_num(subg.edges()[1], tag)
        u, v = subg.edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert len(edge_set) == subg.number_of_edges()

    # outedge / with replacement
    for _ in range(5):
        subg = dgl.sampling.sample_neighbors_biased(g_sorted, g.srcnodes(), 5, bias, edge_dir='out', replace=True)
        check_num(subg.edges()[1], tag)

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sample neighbors not implemented")
def test_sample_neighbors_etype_homogeneous():
    num_nodes = 100
    rare_cnt = 4
    g = create_etype_test_graph(100, 30, rare_cnt)
    h_g = dgl.to_homogeneous(g)
    seed_ntype = g.get_ntype_id("u")
    seeds = F.nonzero_1d(h_g.ndata[dgl.NTYPE] == seed_ntype)

    def check_num(nodes, replace):
        nodes = F.asnumpy(nodes)
        cnt = [sum(nodes == i) for i in range(num_nodes)]

        for i in range(20):
            if i < rare_cnt:
                if replace is False:
                    assert cnt[i] == 22
                else:
                    assert cnt[i] == 30
            else:
                if replace is False:
                    assert cnt[i] == 12
                else:
                    assert cnt[i] == 20

    # graph with coo format
    coo_g = h_g.formats('coo')
    for _ in range(5):
        subg = dgl.sampling.sample_etype_neighbors(coo_g, seeds, dgl.ETYPE, 10, replace=False)
        check_num(subg.edges()[1], False)

    for _ in range(5):
        subg = dgl.sampling.sample_etype_neighbors(coo_g, seeds, dgl.ETYPE, 10, replace=True)
        check_num(subg.edges()[1], True)

    # graph with csr format
    csr_g = h_g.formats('csr')
    csr_g = csr_g.formats(['csr','csc','coo'])
    for _ in range(5):
        subg = dgl.sampling.sample_etype_neighbors(csr_g, seeds, dgl.ETYPE, 10, replace=False)
        check_num(subg.edges()[1], False)

    for _ in range(5):
        subg = dgl.sampling.sample_etype_neighbors(csr_g, seeds, dgl.ETYPE, 10, replace=True)
        check_num(subg.edges()[1], True)

    # graph with csc format
    csc_g = h_g.formats('csc')
    for _ in range(5):
        subg = dgl.sampling.sample_etype_neighbors(csc_g, seeds, dgl.ETYPE, 10, replace=False)
        check_num(subg.edges()[1], False)

    for _ in range(5):
        subg = dgl.sampling.sample_etype_neighbors(csc_g, seeds, dgl.ETYPE, 10, replace=True)
        check_num(subg.edges()[1], True)

    def check_num2(nodes, replace):
        nodes = F.asnumpy(nodes)
        cnt = [sum(nodes == i) for i in range(num_nodes)]

        for i in range(20):
            if replace is False:
                assert cnt[i] == 7
            else:
                assert cnt[i] == 10

    # edge dir out
    # graph with coo format
    coo_g = h_g.formats('coo')
    for _ in range(5):
        subg = dgl.sampling.sample_etype_neighbors(
            coo_g, seeds, dgl.ETYPE, 5, edge_dir='out', replace=False)
        check_num2(subg.edges()[0], False)
    for _ in range(5):
        subg = dgl.sampling.sample_etype_neighbors(
            coo_g, seeds, dgl.ETYPE, 5, edge_dir='out', replace=True)
        check_num2(subg.edges()[0], True)
    # graph with csr format
    csr_g = h_g.formats('csr')
    for _ in range(5):
        subg = dgl.sampling.sample_etype_neighbors(
            csr_g, seeds, dgl.ETYPE, 5, edge_dir='out', replace=False)
        check_num2(subg.edges()[0], False)

    for _ in range(5):
        subg = dgl.sampling.sample_etype_neighbors(
            csr_g, seeds, dgl.ETYPE, 5, edge_dir='out', replace=True)
        check_num2(subg.edges()[0], True)

    # graph with csc format
    csc_g = h_g.formats('csc')
    csc_g = csc_g.formats(['csc','csr','coo'])
    for _ in range(5):
        subg = dgl.sampling.sample_etype_neighbors(
            csc_g, seeds, dgl.ETYPE, 5, edge_dir='out', replace=False)
        check_num2(subg.edges()[0], False)

    for _ in range(5):
        subg = dgl.sampling.sample_etype_neighbors(
            csc_g, seeds, dgl.ETYPE, 5, edge_dir='out', replace=True)
        check_num2(subg.edges()[0], True)


if __name__ == '__main__':
    test_sample_neighbors_etype_homogeneous()
    test_random_walk()
    test_pack_traces()
    test_pinsage_sampling()
    test_sample_neighbors_outedge()
    test_sample_neighbors_topk()
    test_sample_neighbors_topk_outedge()
    test_sample_neighbors_with_0deg()
    test_sample_neighbors_biased_homogeneous()
    test_sample_neighbors_biased_bipartite()
