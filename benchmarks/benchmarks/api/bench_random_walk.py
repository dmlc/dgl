import time

import dgl

import torch

from .. import utils


def _random_walk(g, seeds, length):
    return dgl.sampling.random_walk(g, seeds, length=length)


def _node2vec(g, seeds, length):
    return dgl.sampling.node2vec_random_walk(g, seeds, 1, 1, length)


@utils.skip_if_gpu()
@utils.benchmark("time")
@utils.parametrize("graph_name", ["cora", "livejournal", "friendster"])
@utils.parametrize("num_seeds", [10, 100, 1000])
@utils.parametrize("length", [2, 5, 10, 20])
@utils.parametrize("algorithm", ["_random_walk", "_node2vec"])
def track_time(graph_name, num_seeds, length, algorithm):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, "csr")
    seeds = torch.randint(0, graph.num_nodes(), (num_seeds,))
    print(graph_name, num_seeds, length)
    alg = globals()[algorithm]
    # dry run
    for i in range(5):
        _ = alg(graph, seeds, length=length)

    # timing
    with utils.Timer() as t:
        for i in range(50):
            _ = alg(graph, seeds, length=length)

    return t.elapsed_secs / 50
