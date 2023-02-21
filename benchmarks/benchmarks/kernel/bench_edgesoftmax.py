import time

import dgl

import torch

from .. import utils


# The benchmarks for ops edge_softmax
@utils.benchmark("time", timeout=600)
@utils.parametrize("graph", ["ogbn-arxiv", "reddit", "cora", "pubmed"])
@utils.parametrize("num_heads", [1, 4, 8])
def track_time(graph, num_heads):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph).to(device)
    score = (
        torch.randn((graph.num_edges(), num_heads))
        .requires_grad_(True)
        .float()
        .to(device)
    )

    # dry run
    for i in range(3):
        y = dgl.ops.edge_softmax(graph, score)

    # timing
    with utils.Timer(device) as t:
        for i in range(100):
            y = dgl.ops.edge_softmax(graph, score)

    return t.elapsed_secs / 100
