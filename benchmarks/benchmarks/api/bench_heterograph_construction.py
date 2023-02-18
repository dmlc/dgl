import time

import dgl
import dgl.function as fn

import numpy as np
import torch

from .. import utils


@utils.benchmark("time")
@utils.parametrize("num_relations", [5, 50, 500])
def track_time(num_relations):
    dd = {}
    candidate_edges = [
        dgl.data.CoraGraphDataset(verbose=False)[0].edges(),
        dgl.data.PubmedGraphDataset(verbose=False)[0].edges(),
        dgl.data.CiteseerGraphDataset(verbose=False)[0].edges(),
    ]
    for i in range(num_relations):
        dd[("n1", "e_{}".format(i), "n2")] = candidate_edges[
            i % len(candidate_edges)
        ]

    # dry run
    graph = dgl.heterograph(dd)

    # timing
    with utils.Timer() as t:
        for i in range(3):
            graph = dgl.heterograph(dd)

    return t.elapsed_secs / 3
