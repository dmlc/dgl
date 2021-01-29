
import time
import dgl
import torch
import numpy as np
import dgl.function as fn


from .. import utils


@utils.skip_if_gpu()
@utils.benchmark('time')
@utils.parametrize('size', ["small", "large"])
def track_time(size):
    edge_list = {
        "small": dgl.data.CiteseerGraphDataset(verbose=False)[0].edges(),
        "large": utils.get_livejournal().edges()
    }

    # dry run
    dgl.graph(edge_list[size])

    # timing
    t0 = time.time()
    for i in range(3):
        g = dgl.graph(edge_list[size])
    t1 = time.time()

    return (t1 - t0) / 3
