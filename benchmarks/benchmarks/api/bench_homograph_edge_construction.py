import time

import dgl
import dgl.function as fn

import numpy as np
import torch

from .. import utils


@utils.skip_if_gpu()
@utils.benchmark("time")
@utils.parametrize("size", ["small", "large"])
def track_time(size):
    edge_list = {
        "small": dgl.data.CiteseerGraphDataset(verbose=False)[0].edges(),
        "large": utils.get_livejournal().edges(),
    }

    # dry run
    dgl.graph(edge_list[size])

    # timing
    with utils.Timer() as t:
        for i in range(10):
            g = dgl.graph(edge_list[size])

    return t.elapsed_secs / 10
