import time

import dgl
import dgl.function as fn

import numpy as np
import torch

from .. import utils


@utils.benchmark("time", timeout=600)
@utils.parametrize("num_relations", [5, 50, 500])
@utils.parametrize("format", ["coo", "csr"])
@utils.parametrize("feat_size", [8, 128, 512])
@utils.parametrize("reduce_type", ["u->e"])  # , 'e->u'])
def track_time(num_relations, format, feat_size, reduce_type):
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

    reduce_builtin_dict = {
        "u->e": fn.copy_u("h", "x"),
        # 'e->u': fn.copy_e('h', 'x'),
    }

    # dry run
    for i in range(3):
        graph.apply_edges(reduce_builtin_dict[reduce_type])

    # timing

    with utils.Timer() as t:
        for i in range(10):
            graph.apply_edges(reduce_builtin_dict[reduce_type])

    return t.elapsed_secs / 10
