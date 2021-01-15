import time
import dgl
import torch

from .. import utils

@utils.benchmark('time')
@utils.parametrize('batch_size', [4, 32, 256])
def track_time(batch_size):
    device = utils.get_bench_device()
    ds = dgl.data.QM7bDataset()
    # prepare graph
    graphs = ds[0:batch_size][0]
    bg = dgl.batch(graphs)

    # dry run
    for i in range(10):
        glist = dgl.unbatch(bg)

    # timing
    t0 = time.time()
    for i in range(100):
        glist = dgl.unbatch(bg)
    t1 = time.time()

    return (t1 - t0) / 100
