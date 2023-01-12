import time

import numpy as np
import torch

import dgl
import dgl.function as fn

from .. import utils


@utils.skip_if_gpu()
@utils.benchmark("time")
@utils.parametrize("graph_name", ["livejournal", "reddit"])
@utils.parametrize("format", ["coo"])
@utils.parametrize("seed_egdes_num", [500, 5000, 50000])
def track_time(graph_name, format, seed_egdes_num):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format)
    graph = graph.to(device)

    seed_edges = np.random.randint(0, graph.num_edges(), seed_egdes_num)

    # dry run
    for i in range(3):
        dgl.edge_subgraph(graph, seed_edges)

    # timing

    with utils.Timer() as t:
        for i in range(3):
            dgl.edge_subgraph(graph, seed_edges)

    return t.elapsed_secs / 3
