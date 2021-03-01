import time
import dgl
import torch
import numpy as np

from .. import utils


@utils.benchmark('time', timeout=60)
@utils.parametrize('k', [3, 5, 10])
@utils.parametrize('size', [50, 200, 1000])
@utils.parametrize('dim', [16, 128, 512])
def track_time(size, dim, k):
    device = utils.get_bench_device()
    features = np.random.RandomState(42).randn(size, dim)
    feat = torch.tensor(features, dtype=torch.float, device=device)
    # dry run
    for i in range(3):
        dgl.knn_graph(feat, k)
    # timing
    with utils.Timer() as t:
        for i in range(20):
            dgl.knn_graph(feat, k)

    return t.elapsed_secs / 20
