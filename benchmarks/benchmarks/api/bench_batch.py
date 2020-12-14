import time
import dgl
import torch

from .. import utils

@utils.benchmark('time')
@utils.parametrize('batch_size', [4, 32, 256])
def track_time(batch_size):
    device = utils.get_bench_device()

    # prepare graph
    graphs = []
    for i in range(batch_size):
        u = torch.randint(20, (40,))
        v = torch.randint(20, (40,))
        graphs.append(dgl.graph((u, v)).to(device))

    # dry run
    for i in range(10):
        g = dgl.batch(graphs)

    # timing
    t0 = time.time()
    for i in range(100):
        g = dgl.batch(graphs)
    t1 = time.time()

    return (t1 - t0) / 100
