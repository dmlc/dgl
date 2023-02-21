import time

import dgl

import numpy as np
import torch

from .. import utils


@utils.benchmark("time", timeout=600)
@utils.parametrize_cpu("graph_name", ["cora", "livejournal", "friendster"])
@utils.parametrize_gpu("graph_name", ["cora", "livejournal"])
@utils.parametrize("format", ["coo"])  # csc is not supported
@utils.parametrize("fraction", [0.01, 0.1])
def track_time(graph_name, format, fraction):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format)
    graph = graph.to(device)
    eids = np.random.choice(
        np.arange(graph.num_edges(), dtype=np.int64),
        int(graph.num_edges() * fraction),
    )
    eids = torch.tensor(eids, device=device, dtype=torch.int64)
    # dry run
    for i in range(10):
        out = graph.find_edges(i)
        out = graph.find_edges(
            torch.arange(i * 10, dtype=torch.int64, device=device)
        )

    # timing

    with utils.Timer() as t:
        for i in range(10):
            edges = graph.find_edges(eids)

    return t.elapsed_secs / 10
