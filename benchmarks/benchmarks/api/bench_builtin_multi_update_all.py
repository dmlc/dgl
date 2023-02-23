import time

import dgl
import dgl.function as fn
import numpy as np
import torch

from .. import utils


@utils.benchmark("time", timeout=600)
@utils.parametrize("feat_size", [32, 128, 512])
@utils.parametrize("num_relations", [5, 50, 500])
@utils.parametrize("multi_reduce_type", ["sum", "stack"])
def track_time(feat_size, num_relations, multi_reduce_type):
    device = utils.get_bench_device()
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
    graph = dgl.heterograph(dd)

    graph = graph.to(device)
    graph.nodes["n1"].data["h"] = torch.randn(
        (graph.num_nodes("n1"), feat_size), device=device
    )
    graph.nodes["n2"].data["h"] = torch.randn(
        (graph.num_nodes("n2"), feat_size), device=device
    )

    # dry run
    update_dict = {}
    for i in range(num_relations):
        update_dict["e_{}".format(i)] = (fn.copy_u("h", "m"), fn.sum("m", "h"))
    graph.multi_update_all(update_dict, multi_reduce_type)

    # timing

    with utils.Timer() as t:
        for i in range(3):
            graph.multi_update_all(update_dict, multi_reduce_type)

    return t.elapsed_secs / 3
