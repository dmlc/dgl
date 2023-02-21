import time

import dgl
import dgl.function as fn

import numpy as np
import torch

from .. import utils


@utils.benchmark("time", timeout=7200)
@utils.parametrize("graph_name", ["ogbn-arxiv", "pubmed"])
@utils.parametrize("format", ["coo"])  # only coo supports udf
@utils.parametrize("feat_size", [8, 32, 128, 512])
@utils.parametrize("reduce_type", ["u->e", "u+v"])
def track_time(graph_name, format, feat_size, reduce_type):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format)
    graph = graph.to(device)
    graph.ndata["h"] = torch.randn(
        (graph.num_nodes(), feat_size), device=device
    )

    reduce_udf_dict = {
        "u->e": lambda edges: {"x": edges.src["h"]},
        "u+v": lambda edges: {"x": edges.src["h"] + edges.dst["h"]},
    }

    # dry run
    graph.apply_edges(reduce_udf_dict[reduce_type])

    # timing
    with utils.Timer() as t:
        for i in range(3):
            graph.apply_edges(reduce_udf_dict[reduce_type])

    return t.elapsed_secs / 3
