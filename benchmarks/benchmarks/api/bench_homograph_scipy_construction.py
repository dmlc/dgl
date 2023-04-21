import time

import dgl
import dgl.function as fn

import numpy as np
import torch

from .. import utils


@utils.skip_if_gpu()
@utils.benchmark("time")
@utils.parametrize("size", ["small", "large"])
@utils.parametrize("scipy_format", ["coo", "csr"])
def track_time(size, scipy_format):
    matrix_dict = {
        "small": dgl.data.CiteseerGraphDataset(verbose=False)[0].adj_external(
            scipy_fmt=scipy_format
        ),
        "large": utils.get_livejournal().adj_external(scipy_fmt=scipy_format),
    }

    # dry run
    dgl.from_scipy(matrix_dict[size])

    # timing
    with utils.Timer() as t:
        for i in range(3):
            dgl.from_scipy(matrix_dict[size])

    return t.elapsed_secs / 3
