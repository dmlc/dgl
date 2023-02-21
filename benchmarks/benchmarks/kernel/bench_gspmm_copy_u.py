import time

import dgl

import torch

from .. import utils


def calc_gflops(graph, feat_size, time):
    return round(graph.num_edges() * feat_size / 1000000000 / time, 2)


@utils.benchmark("flops", timeout=600)
@utils.parametrize("graph", ["ogbn-arxiv", "reddit", "ogbn-proteins"])
@utils.parametrize("feat_size", [4, 32, 256])
@utils.parametrize("reducer", ["sum", "max"])
def track_flops(graph, feat_size, reducer):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph, format="csc").to(device)
    x = torch.randn(graph.num_nodes(), feat_size, device=device)

    if reducer == "sum":
        op = dgl.ops.copy_u_sum
    elif reducer == "max":
        op = dgl.ops.copy_u_max
    else:
        raise ValueError("Invalid reducer", reducer)

    # dry run
    for i in range(3):
        y = op(graph, x)

    # timing
    with utils.Timer(device) as t:
        for i in range(10):
            y = op(graph, x)

    return calc_gflops(graph, feat_size, t.elapsed_secs / 10)
