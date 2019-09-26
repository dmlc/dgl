import dgl
from dgl import utils
import backend as F
import numpy as np

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

def test_random_walk_with_restart():
    edge_list = [(0, 1), (1, 2), (2, 3), (3, 4),
                 (4, 3), (3, 2), (2, 1), (1, 0)]
    seeds = [0, 1]
    max_nodes = 10

    g = dgl.DGLGraph(edge_list)

    # test normal RWR
    traces = dgl.contrib.sampling.random_walk_with_restart(g, seeds, 0.2, max_nodes)
    assert len(traces) == len(seeds)
    for traces_per_seed in traces:
        total_nodes = 0
        for t in traces_per_seed:
            total_nodes += len(t)
            trace_diff = np.diff(F.zerocopy_to_numpy(t), axis=-1)
            assert (np.abs(trace_diff) == 1).all()
        assert total_nodes >= max_nodes

    # test RWR with early stopping
    traces = dgl.contrib.sampling.random_walk_with_restart(
            g, seeds, 1, 100, max_nodes, 1)
    assert len(traces) == len(seeds)
    for traces_per_seed in traces:
        assert sum(len(t) for t in traces_per_seed) < 100

    # test bipartite RWR
    traces = dgl.contrib.sampling.bipartite_single_sided_random_walk_with_restart(
            g, seeds, 0.2, max_nodes)
    assert len(traces) == len(seeds)
    for traces_per_seed in traces:
        for t in traces_per_seed:
            trace_diff = np.diff(F.zerocopy_to_numpy(t), axis=-1)
            assert (trace_diff % 2 == 0).all()

def test_metapath_random_walk():
    g1 = dgl.bipartite(([0, 1, 2, 3], [0, 1, 2, 3]), 'a', 'ab', 'b')
    g2 = dgl.bipartite(([0, 0, 1, 1, 2, 2, 3, 3], [1, 3, 2, 0, 3, 1, 0, 2]), 'b', 'ba', 'a')
    G = dgl.hetero_from_relations([g1, g2])
    seeds = [0, 1]
    traces = dgl.contrib.sampling.metapath_random_walk(G, ['ab', 'ba'] * 4, seeds, 3)
    for seed, traces_per_seed in zip(seeds, traces):
        assert len(traces_per_seed) == 3
        for trace in traces_per_seed:
            assert len(trace) == 8
            trace = np.insert(F.asnumpy(trace), 0, seed)
            for i in range(4):
                assert g1.has_edge_between(trace[2 * i], trace[2 * i + 1])
                assert g2.has_edge_between(trace[2 * i + 1], trace[2 * i + 2])

if __name__ == '__main__':
    test_random_walk()
    test_metapath_random_walk()
