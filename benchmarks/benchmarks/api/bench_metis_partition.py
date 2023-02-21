import time

import dgl

import numpy as np
import torch

from .. import utils


@utils.skip_if_gpu()
@utils.benchmark("time", timeout=1200)
@utils.parametrize("graph_name", ["reddit"])
@utils.parametrize("k", [2, 4, 8])
def track_time(graph_name, k):
    device = utils.get_bench_device()
    data = utils.process_data(graph_name)
    graph = data[0]
    # dry run
    gg = dgl.transforms.metis_partition(graph, k)

    # timing
    with utils.Timer() as t:
        for i in range(3):
            gg = dgl.transforms.metis_partition(graph, k)

    return t.elapsed_secs / 3
