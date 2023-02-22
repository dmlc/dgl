import time

import dgl
import dgl.function as fn

import numpy as np
import torch

from .. import utils


@utils.benchmark("time", timeout=600)
@utils.parametrize("graph_name", ["cora", "ogbn-arxiv"])
@utils.parametrize("format", ["coo", "csr"])
@utils.parametrize("feat_size", [8, 128, 512])
@utils.parametrize("reduce_type", ["u->e", "u+v"])
def track_time(graph_name, format, feat_size, reduce_type):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format)
    graph = graph.to(device)
    graph.ndata["h"] = torch.randn(
        (graph.num_nodes(), feat_size), device=device
    )

    reduce_builtin_dict = {
        "u->e": fn.copy_u("h", "x"),
        "u+v": fn.u_add_v("h", "h", "x"),
    }

    # dry run
    for i in range(3):
        graph.apply_edges(reduce_builtin_dict[reduce_type])

    # timing

    with utils.Timer() as t:
        for i in range(10):
            graph.apply_edges(reduce_builtin_dict[reduce_type])

    return t.elapsed_secs / 10
