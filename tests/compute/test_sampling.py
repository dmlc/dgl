import dgl
import backend as F
import numpy as np
import unittest
from collections import defaultdict
import pytest

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

@pytest.mark.parametrize('use_uva', [True, False])
def test_non_uniform_random_walk(use_uva):
    if use_uva:
        if F.ctx() == F.cpu():
            pytest.skip('UVA biased random walk requires a GPU.')
        if dgl.backend.backend_name != 'pytorch':
            pytest.skip('UVA biased random walk is only supported with PyTorch.')
    g2 = dgl.heterograph({
            ('user', 'follow', 'user'): ([0, 1, 1, 2, 3], [1, 2, 3, 0, 0])
        })
    g4 = dgl.heterograph({
            ('user', 'follow', 'user'): ([0, 1, 1, 2, 3], [1, 2, 3, 0, 0]),
            ('user', 'view', 'item'): ([0, 0, 1, 2, 3, 3], [0, 1, 1, 2, 2, 1]),
            ('item', 'viewed-by', 'user'): ([0, 1, 1, 2, 2, 1], [0, 0, 1, 2, 3, 3])
        })

    g2.edata['p'] = F.copy_to(F.tensor([3, 0, 3, 3, 3], dtype=F.float32), F.cpu())
    g2.edata['p2'] = F.copy_to(F.tensor([[3], [0], [3], [3], [3]], dtype=F.float32), F.cpu())
    g4.edges['follow'].data['p'] = F.copy_to(F.tensor([3, 0, 3, 3, 3], dtype=F.float32), F.cpu())
    g4.edges['viewed-by'].data['p'] = F.copy_to(F.tensor([1, 1, 1, 1, 1, 1], dtype=F.float32), F.cpu())

    if use_uva:
        for g in (g2, g4):
            g.create_formats_()
            g.pin_memory_()
    elif F._default_context_str == 'gpu':
        g2 = g2.to(F.ctx())
        g4 = g4.to(F.ctx())

    try:
        traces, eids, ntypes = dgl.sampling.random_walk(
            g2, F.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=g2.idtype),
            length=4, prob='p', return_eids=True)
        check_random_walk(g2, ['follow'] * 4, traces, ntypes, 'p', trace_eids=eids)

        with pytest.raises(dgl.DGLError):
            traces, ntypes = dgl.sampling.random_walk(
                g2, F.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=g2.idtype),
                length=4, prob='p2')

        metapath = ['follow', 'view', 'viewed-by'] * 2
        traces, eids, ntypes = dgl.sampling.random_walk(
            g4, F.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=g4.idtype),
            metapath=metapath, prob='p', return_eids=True)
        check_random_walk(g4, metapath, traces, ntypes, 'p', trace_eids=eids)
        traces, eids, ntypes = dgl.sampling.random_walk(
            g4, F.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=g4.idtype),
            metapath=metapath, prob='p', restart_prob=0., return_eids=True)
        check_random_walk(g4, metapath, traces, ntypes, 'p', trace_eids=eids)
        traces, eids, ntypes = dgl.sampling.random_walk(
            g4, F.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=g4.idtype),
            metapath=metapath, prob='p',
            restart_prob=F.zeros((6,), F.float32, F.ctx()), return_eids=True)
        check_random_walk(g4, metapath, traces, ntypes, 'p', trace_eids=eids)
        traces, eids, ntypes = dgl.sampling.random_walk(
            g4, F.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=g4.idtype),
            metapath=metapath + ['follow'], prob='p',
            restart_prob=F.tensor([0, 0, 0, 0, 0, 0, 1], F.float32), return_eids=True)
        check_random_walk(g4, metapath, traces[:, :7], ntypes[:7], 'p', trace_eids=eids)
        assert (F.asnumpy(traces[:, 7]) == -1).all()
    finally:
        for g in (g2, g4):
            g.unpin_memory_()

@pytest.mark.parametrize('use_uva', [True, False])
def test_uniform_random_walk(use_uva):
    if use_uva and F.ctx() == F.cpu():
        pytest.skip('UVA random walk requires a GPU.')
    g1 = dgl.heterograph({
            ('user', 'follow', 'user'): ([0, 1, 2], [1, 2, 0])
        })
    g2 = dgl.heterograph({
            ('user', 'follow', 'user'): ([0, 1, 1, 2, 3], [1, 2, 3, 0, 0])
        })
    g3 = dgl.heterograph({
            ('user', 'follow', 'user'): ([0, 1, 2], [1, 2, 0]),
            ('user', 'view', 'item'): ([0, 1, 2], [0, 1, 2]),
            ('item', 'viewed-by', 'user'): ([0, 1, 2], [0, 1, 2])
        })
    g4 = dgl.heterograph({
            ('user', 'follow', 'user'): ([0, 1, 1, 2, 3], [1, 2, 3, 0, 0]),
            ('user', 'view', 'item'): ([0, 0, 1, 2, 3, 3], [0, 1, 1, 2, 2, 1]),
            ('item', 'viewed-by', 'user'): ([0, 1, 1, 2, 2, 1], [0, 0, 1, 2, 3, 3])
        })

    if use_uva:
        for g in (g1, g2, g3, g4):
            g.create_formats_()
            g.pin_memory_()
    elif F._default_context_str == 'gpu':
        g1 = g1.to(F.ctx())
        g2 = g2.to(F.ctx())
        g3 = g3.to(F.ctx())
        g4 = g4.to(F.ctx())

    try:
        traces, eids, ntypes = dgl.sampling.random_walk(
            g1, F.tensor([0, 1, 2, 0, 1, 2], dtype=g1.idtype), length=4, return_eids=True)
        check_random_walk(g1, ['follow'] * 4, traces, ntypes, trace_eids=eids)
        if F._default_context_str == 'cpu':
            with pytest.raises(dgl.DGLError):
                dgl.sampling.random_walk(g1, F.tensor([0, 1, 2, 10], dtype=g1.idtype), length=4, return_eids=True)
        traces, eids, ntypes = dgl.sampling.random_walk(
            g1, F.tensor([0, 1, 2, 0, 1, 2], dtype=g1.idtype), length=4, restart_prob=0., return_eids=True)
        check_random_walk(g1, ['follow'] * 4, traces, ntypes, trace_eids=eids)
        traces, ntypes = dgl.sampling.random_walk(
            g1, F.tensor([0, 1, 2, 0, 1, 2], dtype=g1.idtype), length=4, restart_prob=F.zeros((4,), F.float32))
        check_random_walk(g1, ['follow'] * 4, traces, ntypes)
        traces, ntypes = dgl.sampling.random_walk(
            g1, F.tensor([0, 1, 2, 0, 1, 2], dtype=g1.idtype), length=5,
            restart_prob=F.tensor([0, 0, 0, 0, 1], dtype=F.float32))
        check_random_walk(
            g1, ['follow'] * 4, F.slice_axis(traces, 1, 0, 5), F.slice_axis(ntypes, 0, 0, 5))
        assert (F.asnumpy(traces)[:, 5] == -1).all()

        traces, eids, ntypes = dgl.sampling.random_walk(
            g2, F.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=g2.idtype), length=4, return_eids=True)
        check_random_walk(g2, ['follow'] * 4, traces, ntypes, trace_eids=eids)

        metapath = ['follow', 'view', 'viewed-by'] * 2
        traces, eids, ntypes = dgl.sampling.random_walk(
            g3, F.tensor([0, 1, 2, 0, 1, 2], dtype=g3.idtype), metapath=metapath, return_eids=True)
        check_random_walk(g3, metapath, traces, ntypes, trace_eids=eids)

        metapath = ['follow', 'view', 'viewed-by'] * 2
        traces, eids, ntypes = dgl.sampling.random_walk(
            g4, F.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=g4.idtype), metapath=metapath, return_eids=True)
        check_random_walk(g4, metapath, traces, ntypes, trace_eids=eids)

        traces, eids, ntypes = dgl.sampling.random_walk(
            g4, F.tensor([0, 1, 2, 0, 1, 2], dtype=g4.idtype), metapath=metapath, return_eids=True)
        check_random_walk(g4, metapath, traces, ntypes, trace_eids=eids)
    finally:    # make sure to unpin the graphs even if some test fails
        for g in (g1, g2, g3, g4):
            if g.is_pinned():
                g.unpin_memory_()

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

@pytest.mark.parametrize('use_uva', [True, False])
def test_pinsage_sampling(use_uva):
    if use_uva and F.ctx() == F.cpu():
        pytest.skip('UVA sampling requires a GPU.')
    def _test_sampler(g, sampler, ntype):
        seeds = F.copy_to(F.tensor([0, 2], dtype=g.idtype), F.ctx())
        neighbor_g = sampler(seeds)
        assert neighbor_g.ntypes == [ntype]
        u, v = neighbor_g.all_edges(form='uv', order='eid')
        uv = list(zip(F.asnumpy(u).tolist(), F.asnumpy(v).tolist()))
        assert (1, 0) in uv or (0, 0) in uv
        assert (2, 2) in uv or (3, 2) in uv

    g = dgl.heterograph({
        ('item', 'bought-by', 'user'): ([0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 2, 3, 2, 3]),
        ('user', 'bought', 'item'): ([0, 1, 0, 1, 2, 3, 2, 3], [0, 0, 1, 1, 2, 2, 3, 3])})
    if use_uva:
        g.create_formats_()
        g.pin_memory_()
    elif F._default_context_str == 'gpu':
        g = g.to(F.ctx())
    try:
        sampler = dgl.sampling.PinSAGESampler(g, 'item', 'user', 4, 0.5, 3, 2)
        _test_sampler(g, sampler, 'item')
        sampler = dgl.sampling.RandomWalkNeighborSampler(g, 4, 0.5, 3, 2, ['bought-by', 'bought'])
        _test_sampler(g, sampler, 'item')
        sampler = dgl.sampling.RandomWalkNeighborSampler(g, 4, 0.5, 3, 2,
            [('item', 'bought-by', 'user'), ('user', 'bought', 'item')])
        _test_sampler(g, sampler, 'item')
    finally:
        if g.is_pinned():
            g.unpin_memory_()

    g = dgl.graph(([0, 0, 1, 1, 2, 2, 3, 3],
                   [0, 1, 0, 1, 2, 3, 2, 3]))
    if use_uva:
        g.create_formats_()
        g.pin_memory_()
    elif F._default_context_str == 'gpu':
        g = g.to(F.ctx())
    try:
        sampler = dgl.sampling.RandomWalkNeighborSampler(g, 4, 0.5, 3, 2)
        _test_sampler(g, sampler, g.ntypes[0])
    finally:
        if g.is_pinned():
            g.unpin_memory_()

    g = dgl.heterograph({
        ('A', 'AB', 'B'): ([0, 2], [1, 3]),
        ('B', 'BC', 'C'): ([1, 3], [2, 1]),
        ('C', 'CA', 'A'): ([2, 1], [0, 2])})
    if use_uva:
        g.create_formats_()
        g.pin_memory_()
    elif F._default_context_str == 'gpu':
        g = g.to(F.ctx())
    try:
        sampler = dgl.sampling.RandomWalkNeighborSampler(g, 4, 0.5, 3, 2, ['AB', 'BC', 'CA'])
        _test_sampler(g, sampler, 'A')
    finally:
        if g.is_pinned():
            g.unpin_memory_()

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

def test_sample_neighbors_prob():
    _test_sample_neighbors(False, 'prob')
    #_test_sample_neighbors(True)

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

def test_sample_neighbors_with_0deg():
    g = dgl.graph(([], []), num_nodes=5).to(F.ctx())
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
@pytest.mark.parametrize('format_', ['coo', 'csr', 'csc'])
@pytest.mark.parametrize('direction', ['in', 'out'])
@pytest.mark.parametrize('replace', [False, True])
def test_sample_neighbors_etype_homogeneous(format_, direction, replace):
    num_nodes = 100
    rare_cnt = 4
    g = create_etype_test_graph(100, 30, rare_cnt)
    h_g = dgl.to_homogeneous(g)
    seed_ntype = g.get_ntype_id("u")
    seeds = F.nonzero_1d(h_g.ndata[dgl.NTYPE] == seed_ntype)
    fanouts = F.tensor([6, 5, 4, 3, 2], dtype=F.int64)

    def check_num(h_g, all_src, all_dst, subg, replace, fanouts, direction):
        src, dst = subg.edges()
        num_etypes = F.asnumpy(h_g.edata[dgl.ETYPE]).max()
        etype_array = F.asnumpy(subg.edata[dgl.ETYPE])
        src = F.asnumpy(src)
        dst = F.asnumpy(dst)
        fanouts = F.asnumpy(fanouts)

        all_etype_array = F.asnumpy(h_g.edata[dgl.ETYPE])
        all_src = F.asnumpy(all_src)
        all_dst = F.asnumpy(all_dst)

        src_per_etype = []
        dst_per_etype = []
        for etype in range(num_etypes):
            src_per_etype.append(src[etype_array == etype])
            dst_per_etype.append(dst[etype_array == etype])

        if replace:
            if direction == 'in':
                in_degree_per_etype = [np.bincount(d) for d in dst_per_etype]
                for in_degree, fanout in zip(in_degree_per_etype, fanouts):
                    assert np.all(in_degree == fanout)
            else:
                out_degree_per_etype = [np.bincount(s) for s in src_per_etype]
                for out_degree, fanout in zip(out_degree_per_etype, fanouts):
                    assert np.all(out_degree == fanout)
        else:
            if direction == 'in':
                for v in set(dst):
                    u = src[dst == v]
                    et = etype_array[dst == v]
                    all_u = all_src[all_dst == v]
                    all_et = all_etype_array[all_dst == v]
                    for etype in set(et):
                        u_etype = set(u[et == etype])
                        all_u_etype = set(all_u[all_et == etype])
                        assert (len(u_etype) == fanouts[etype]) or (u_etype == all_u_etype)
            else:
                for u in set(src):
                    v = dst[src == u]
                    et = etype_array[src == u]
                    all_v = all_dst[all_src == u]
                    all_et = all_etype_array[all_src == u]
                    for etype in set(et):
                        v_etype = set(v[et == etype])
                        all_v_etype = set(all_v[all_et == etype])
                        assert (len(v_etype) == fanouts[etype]) or (v_etype == all_v_etype)

    all_src, all_dst = h_g.edges()
    h_g = h_g.formats(format_)
    if (direction, format_) in [('in', 'csr'), ('out', 'csc')]:
        h_g = h_g.formats(['csc', 'csr', 'coo'])
    for _ in range(5):
        subg = dgl.sampling.sample_etype_neighbors(
            h_g, seeds, dgl.ETYPE, fanouts, replace=replace, edge_dir=direction)
        check_num(h_g, all_src, all_dst, subg, replace, fanouts, direction)


@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sample neighbors not implemented")
@pytest.mark.parametrize('format_', ['csr', 'csc'])
@pytest.mark.parametrize('direction', ['in', 'out'])
def test_sample_neighbors_etype_sorted_homogeneous(format_, direction):
    rare_cnt = 4
    g = create_etype_test_graph(100, 30, rare_cnt)
    h_g = dgl.to_homogeneous(g)
    seed_ntype = g.get_ntype_id("u")
    seeds = F.nonzero_1d(h_g.ndata[dgl.NTYPE] == seed_ntype)
    fanouts = F.tensor([6, 5, -1, 3, 2], dtype=F.int64)
    h_g = h_g.formats(format_)
    if (direction, format_) in [('in', 'csr'), ('out', 'csc')]:
        h_g = h_g.formats(['csc', 'csr', 'coo'])
    orig_etype = F.asnumpy(h_g.edata[dgl.ETYPE])
    h_g.edata[dgl.ETYPE] = F.tensor(
        np.sort(orig_etype)[::-1].tolist(), dtype=F.int64)

    try:
        dgl.sampling.sample_etype_neighbors(
            h_g, seeds, dgl.ETYPE, fanouts, edge_dir=direction, etype_sorted=True)
        fail = False
    except dgl.DGLError:
        fail = True
    assert fail

@pytest.mark.parametrize('dtype', ['int32', 'int64'])
def test_sample_neighbors_exclude_edges_heteroG(dtype):
    d_i_d_u_nodes = F.zerocopy_from_numpy(np.unique(np.random.randint(300, size=100, dtype=dtype)))
    d_i_d_v_nodes = F.zerocopy_from_numpy(np.random.randint(25, size=d_i_d_u_nodes.shape, dtype=dtype))
    d_i_g_u_nodes = F.zerocopy_from_numpy(np.unique(np.random.randint(300, size=100, dtype=dtype)))
    d_i_g_v_nodes = F.zerocopy_from_numpy(np.random.randint(25, size=d_i_g_u_nodes.shape, dtype=dtype))
    d_t_d_u_nodes = F.zerocopy_from_numpy(np.unique(np.random.randint(300, size=100, dtype=dtype)))
    d_t_d_v_nodes = F.zerocopy_from_numpy(np.random.randint(25, size=d_t_d_u_nodes.shape, dtype=dtype))

    g = dgl.heterograph({
        ('drug', 'interacts', 'drug'): (d_i_d_u_nodes, d_i_d_v_nodes),
        ('drug', 'interacts', 'gene'): (d_i_g_u_nodes, d_i_g_v_nodes),
        ('drug', 'treats', 'disease'): (d_t_d_u_nodes, d_t_d_v_nodes)
    }).to(F.ctx())

    (U, V, EID) = (0, 1, 2)

    nd_b_idx = np.random.randint(low=1, high=24, dtype=dtype)
    nd_e_idx = np.random.randint(low=25, high=49, dtype=dtype)
    did_b_idx = np.random.randint(low=1, high=24, dtype=dtype)
    did_e_idx = np.random.randint(low=25, high=49, dtype=dtype)
    sampled_amount = np.random.randint(low=1, high=10, dtype=dtype)

    drug_i_drug_edges = g.all_edges(form='all', etype=('drug','interacts','drug'))
    excluded_d_i_d_edges = drug_i_drug_edges[EID][did_b_idx:did_e_idx]
    sampled_drug_node = drug_i_drug_edges[V][nd_b_idx:nd_e_idx]
    did_excluded_nodes_U = drug_i_drug_edges[U][did_b_idx:did_e_idx]
    did_excluded_nodes_V = drug_i_drug_edges[V][did_b_idx:did_e_idx]

    nd_b_idx = np.random.randint(low=1, high=24, dtype=dtype)
    nd_e_idx = np.random.randint(low=25, high=49, dtype=dtype)
    dig_b_idx = np.random.randint(low=1, high=24, dtype=dtype)
    dig_e_idx = np.random.randint(low=25, high=49, dtype=dtype)
    drug_i_gene_edges = g.all_edges(form='all', etype=('drug','interacts','gene'))
    excluded_d_i_g_edges = drug_i_gene_edges[EID][dig_b_idx:dig_e_idx]
    dig_excluded_nodes_U = drug_i_gene_edges[U][dig_b_idx:dig_e_idx]
    dig_excluded_nodes_V = drug_i_gene_edges[V][dig_b_idx:dig_e_idx]
    sampled_gene_node = drug_i_gene_edges[V][nd_b_idx:nd_e_idx]

    nd_b_idx = np.random.randint(low=1, high=24, dtype=dtype)
    nd_e_idx = np.random.randint(low=25, high=49, dtype=dtype)
    dtd_b_idx = np.random.randint(low=1, high=24, dtype=dtype)
    dtd_e_idx = np.random.randint(low=25, high=49, dtype=dtype)
    drug_t_dis_edges = g.all_edges(form='all', etype=('drug','treats','disease'))
    excluded_d_t_d_edges = drug_t_dis_edges[EID][dtd_b_idx:dtd_e_idx]
    dtd_excluded_nodes_U = drug_t_dis_edges[U][dtd_b_idx:dtd_e_idx]
    dtd_excluded_nodes_V = drug_t_dis_edges[V][dtd_b_idx:dtd_e_idx]
    sampled_disease_node = drug_t_dis_edges[V][nd_b_idx:nd_e_idx]
    excluded_edges  = {('drug', 'interacts', 'drug'): excluded_d_i_d_edges,
                       ('drug', 'interacts', 'gene'): excluded_d_i_g_edges,
                       ('drug', 'treats', 'disease'): excluded_d_t_d_edges
                      }

    sg = dgl.sampling.sample_neighbors(g, {'drug': sampled_drug_node,
                                           'gene': sampled_gene_node,
                                           'disease': sampled_disease_node},
                                       sampled_amount, exclude_edges=excluded_edges)

    assert not np.any(F.asnumpy(sg.has_edges_between(did_excluded_nodes_U,did_excluded_nodes_V,
                                                     etype=('drug','interacts','drug'))))
    assert not np.any(F.asnumpy(sg.has_edges_between(dig_excluded_nodes_U,dig_excluded_nodes_V,
                                                     etype=('drug','interacts','gene'))))
    assert not np.any(F.asnumpy(sg.has_edges_between(dtd_excluded_nodes_U,dtd_excluded_nodes_V,
                                                     etype=('drug','treats','disease'))))

@pytest.mark.parametrize('dtype', ['int32', 'int64'])
def test_sample_neighbors_exclude_edges_homoG(dtype):
    u_nodes = F.zerocopy_from_numpy(np.unique(np.random.randint(300,size=100, dtype=dtype)))
    v_nodes = F.zerocopy_from_numpy(np.random.randint(25, size=u_nodes.shape, dtype=dtype))
    g = dgl.graph((u_nodes, v_nodes)).to(F.ctx())

    (U, V, EID) = (0, 1, 2)

    nd_b_idx = np.random.randint(low=1,high=24, dtype=dtype)
    nd_e_idx = np.random.randint(low=25,high=49, dtype=dtype)
    b_idx = np.random.randint(low=1,high=24, dtype=dtype)
    e_idx = np.random.randint(low=25,high=49, dtype=dtype)
    sampled_amount = np.random.randint(low=1,high=10, dtype=dtype)

    g_edges = g.all_edges(form='all')
    excluded_edges = g_edges[EID][b_idx:e_idx]
    sampled_node = g_edges[V][nd_b_idx:nd_e_idx]
    excluded_nodes_U = g_edges[U][b_idx:e_idx]
    excluded_nodes_V = g_edges[V][b_idx:e_idx]

    sg = dgl.sampling.sample_neighbors(g, sampled_node,
                                       sampled_amount, exclude_edges=excluded_edges)

    assert not np.any(F.asnumpy(sg.has_edges_between(excluded_nodes_U,excluded_nodes_V)))

@pytest.mark.parametrize('dtype', ['int32', 'int64'])
def test_global_uniform_negative_sampling(dtype):
    g = dgl.graph(([], []), num_nodes=1000).to(F.ctx())
    src, dst = dgl.sampling.global_uniform_negative_sampling(g, 2000, False, True)
    assert len(src) == 2000
    assert len(dst) == 2000

    g = dgl.graph((np.random.randint(0, 20, (300,)), np.random.randint(0, 20, (300,)))).to(F.ctx())
    src, dst = dgl.sampling.global_uniform_negative_sampling(g, 20, False, True)
    assert not F.asnumpy(g.has_edges_between(src, dst)).any()

    src, dst = dgl.sampling.global_uniform_negative_sampling(g, 20, False, False)
    assert not F.asnumpy(g.has_edges_between(src, dst)).any()
    src = F.asnumpy(src)
    dst = F.asnumpy(dst)
    s = set(zip(src.tolist(), dst.tolist()))
    assert len(s) == len(src)

    g = dgl.graph(([0], [1])).to(F.ctx())
    src, dst = dgl.sampling.global_uniform_negative_sampling(g, 20, True, False, redundancy=10)
    src = F.asnumpy(src)
    dst = F.asnumpy(dst)
    # should have either no element or (1, 0)
    assert len(src) < 2
    assert len(dst) < 2
    if len(src) == 1:
        assert src[0] == 1
        assert dst[0] == 0

    g = dgl.heterograph({
        ('A', 'AB', 'B'): (np.random.randint(0, 20, (300,)), np.random.randint(0, 40, (300,))),
        ('B', 'BA', 'A'): (np.random.randint(0, 40, (200,)), np.random.randint(0, 20, (200,)))}).to(F.ctx())
    src, dst = dgl.sampling.global_uniform_negative_sampling(g, 20, False, etype='AB')
    assert not F.asnumpy(g.has_edges_between(src, dst, etype='AB')).any()


if __name__ == '__main__':
    from itertools import product
    for args in product(['coo', 'csr', 'csc'], ['in', 'out'], [False, True]):
        test_sample_neighbors_etype_homogeneous(*args)
    test_non_uniform_random_walk()
    test_uniform_random_walk(False)
    test_pack_traces()
    test_pinsage_sampling()
    test_sample_neighbors_outedge()
    test_sample_neighbors_topk()
    test_sample_neighbors_topk_outedge()
    test_sample_neighbors_with_0deg()
    test_sample_neighbors_biased_homogeneous()
    test_sample_neighbors_biased_bipartite()
    test_sample_neighbors_exclude_edges_heteroG('int32')
    test_sample_neighbors_exclude_edges_homoG('int32')
    test_global_uniform_negative_sampling('int32')
    test_global_uniform_negative_sampling('int64')
