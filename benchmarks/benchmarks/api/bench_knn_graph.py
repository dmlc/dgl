import time

import dgl

import numpy as np
import torch

from .. import utils


@utils.benchmark("time", timeout=60)
@utils.parametrize("k", [8, 64])
@utils.parametrize("size", [1000, 10000])
@utils.parametrize("dim", [4, 32, 256])
@utils.parametrize_cpu(
    "algorithm", ["bruteforce-blas", "bruteforce", "kd-tree", "nn-descent"]
)
@utils.parametrize_gpu(
    "algorithm",
    ["bruteforce-blas", "bruteforce", "bruteforce-sharemem", "nn-descent"],
)
def track_time(size, dim, k, algorithm):
    device = utils.get_bench_device()
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
