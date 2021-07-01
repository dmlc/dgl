import time
import dgl
import torch
import numpy as np

from .. import utils

@utils.benchmark('time', timeout=240)
@utils.parametrize('k', [8, 64])
@utils.parametrize('size', [1000, 10000])
@utils.parametrize('dim', [3, 64, 128])
@utils.parametrize('algorithm', ['bruteforce-blas', 'bruteforce', 'kd-tree', 'bruteforce-sharemem', 'nn-descent'])
def track_time(size, dim, k, algorithm):
    device = utils.get_bench_device()
    # skip unavailable pairs of (algorithms, devices)
    if device is 'cuda' and algorithm=='kd-tree':
        return 0
    if device is 'cpu' and algorithm=='bruteforce-sharemem':
        return 0
    features = np.random.RandomState(42).randn(size, dim)
    feat = torch.tensor(features, dtype=torch.float, device=device)
    # dry run
    for i in range(1):
        dgl.knn_graph(feat, k, algorithm=algorithm)
    # timing
    with utils.Timer() as t:
        for i in range(5):
            dgl.knn_graph(feat, k, algorithm=algorithm)

    return t.elapsed_secs / 5
