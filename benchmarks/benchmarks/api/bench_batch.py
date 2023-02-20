import time

import dgl

import torch

from .. import utils


@utils.benchmark("time")
@utils.parametrize("batch_size", [4, 32, 256, 1024])
def track_time(batch_size):
    device = utils.get_bench_device()
    ds = dgl.data.QM7bDataset()
    # prepare graph
    graphs = []
    for graph in ds[0:batch_size][0]:
        g = graph.to(device)
        graphs.append(g)

    # dry run
    for i in range(10):
        g = dgl.batch(graphs)

    # timing

    with utils.Timer() as t:
        for i in range(100):
            g = dgl.batch(graphs)

    return t.elapsed_secs / 100
