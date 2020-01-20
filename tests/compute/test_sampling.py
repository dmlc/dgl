import dgl
import backend as F
import numpy as np
import unittest

def check_random_walk(g, metapath, traces, ntypes, prob=None):
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
                eids = g.edge_id(traces[i, j], traces[i, j+1], etype=metapath[j])
                assert p[eids] != 0

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU random walk not implemented")
def test_random_walk():
    g1 = dgl.heterograph({
        ('user', 'follow', 'user'): [(0, 1), (1, 2), (2, 0)]
        })
    g2 = dgl.heterograph({
        ('user', 'follow', 'user'): [(0, 1), (1, 2), (1, 3), (2, 0), (3, 0)]
        })
    g3 = dgl.heterograph({
        ('user', 'follow', 'user'): [(0, 1), (1, 2), (2, 0)],
        ('user', 'view', 'item'): [(0, 0), (1, 1), (2, 2)],
        ('item', 'viewed-by', 'user'): [(0, 0), (1, 1), (2, 2)]})
    g4 = dgl.heterograph({
        ('user', 'follow', 'user'): [(0, 1), (1, 2), (1, 3), (2, 0), (3, 0)],
        ('user', 'view', 'item'): [(0, 0), (0, 1), (1, 1), (2, 2), (3, 2), (3, 1)],
        ('item', 'viewed-by', 'user'): [(0, 0), (1, 0), (1, 1), (2, 2), (2, 3), (1, 3)]})

    g2.edata['p'] = F.tensor([3, 0, 3, 3, 3], dtype=F.float32)
    g4.edges['follow'].data['p'] = F.tensor([3, 0, 3, 3, 3], dtype=F.float32)
    g4.edges['viewed-by'].data['p'] = F.tensor([1, 1, 1, 1, 1, 1], dtype=F.float32)

    traces, ntypes = dgl.sampling.random_walk(g1, [0, 1, 2, 0, 1, 2], length=4)
    check_random_walk(g1, ['follow'] * 4, traces, ntypes)

    traces, ntypes = dgl.sampling.random_walk(
        g2, [0, 1, 2, 3, 0, 1, 2, 3], length=4)
    check_random_walk(g2, ['follow'] * 4, traces, ntypes)

    traces, ntypes = dgl.sampling.random_walk(
        g2, [0, 1, 2, 3, 0, 1, 2, 3], length=4, prob='p')
    check_random_walk(g2, ['follow'] * 4, traces, ntypes, 'p')

    metapath = ['follow', 'view', 'viewed-by'] * 2
    traces, ntypes = dgl.sampling.random_walk(
        g3, [0, 1, 2, 0, 1, 2], metapath=metapath)
    check_random_walk(g3, metapath, traces, ntypes)

    metapath = ['follow', 'view', 'viewed-by'] * 2
    traces, ntypes = dgl.sampling.random_walk(
        g4, [0, 1, 2, 3, 0, 1, 2, 3], metapath=metapath)
    check_random_walk(g4, metapath, traces, ntypes)

    metapath = ['follow', 'view', 'viewed-by'] * 2
    traces, ntypes = dgl.sampling.random_walk(
        g4, [0, 1, 2, 3, 0, 1, 2, 3], metapath=metapath, prob='p')
    check_random_walk(g4, metapath, traces, ntypes, 'p')


if __name__ == '__main__':
    test_random_walk()
