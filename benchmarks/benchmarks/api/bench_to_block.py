import time

import dgl

import numpy as np
import torch

from .. import utils


@utils.skip_if_gpu()
@utils.benchmark("time", timeout=1200)
@utils.parametrize("graph_name", ["reddit", "ogbn-products"])
@utils.parametrize("num_seed_nodes", [32, 256, 1024, 2048])
@utils.parametrize("fanout", [5, 10, 20])
def track_time(graph_name, num_seed_nodes, fanout):
    device = utils.get_bench_device()
    data = utils.process_data(graph_name)
    graph = data[0]

    # dry run
    dgl.sampling.sample_neighbors(graph, [1, 2, 3], fanout)

    subg_list = []
    for i in range(10):
        seed_nodes = np.random.randint(
            0, graph.num_nodes(), size=num_seed_nodes
        )
        subg = dgl.sampling.sample_neighbors(graph, seed_nodes, fanout)
        subg_list.append(subg)

    # timing
    with utils.Timer() as t:
        for i in range(10):
            gg = dgl.to_block(subg_list[i])

    return t.elapsed_secs / 10
