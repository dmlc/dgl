import time

import dgl

import numpy as np
import torch

from .. import utils


@utils.benchmark("time", timeout=60)
@utils.parametrize("graph_name", ["cora"])
@utils.parametrize("format", ["coo", "csr"])
@utils.parametrize("k", [1, 3, 5])
def track_time(graph_name, format, k):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format)
    graph = graph.to(device)
    graph = graph.formats([format])
    # dry run
    dgl.khop_graph(graph, k)

    # timing
    with utils.Timer() as t:
        for i in range(10):
            gg = dgl.khop_graph(graph, k)

    return t.elapsed_secs / 10
