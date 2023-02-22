import time

import dgl
import dgl.function as fn

import numpy as np
import torch

from .. import utils


@utils.benchmark("time")
@utils.parametrize("graph_name", ["livejournal", "reddit"])
@utils.parametrize("format", ["coo"])
@utils.parametrize("seed_egdes_num", [500, 5000, 50000])
def track_time(graph_name, format, seed_egdes_num):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format)
    graph = graph.to(device)

    seed_edges = np.random.randint(0, graph.num_edges(), seed_egdes_num)
    seed_edges = torch.from_numpy(seed_edges).to(device)

    # dry run
    for i in range(3):
        dgl.edge_subgraph(graph, seed_edges)

    # timing
    num_iters = 50
    with utils.Timer() as t:
        for i in range(num_iters):
            dgl.edge_subgraph(graph, seed_edges)

    return t.elapsed_secs / num_iters
