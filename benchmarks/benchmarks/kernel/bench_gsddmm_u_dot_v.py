import time

import dgl

import torch

from .. import utils


def calc_gflops(graph, feat_size, num_heads, time):
    return round(
        2 * graph.num_edges() * feat_size / 1000000000 / time, 2
    )  # count both mul and add


# The benchmarks include broadcasting cases.
# Given feat_size = D, num_heads = H, the node feature shape will be (H, D // H)
#   while the edge feature shape will be (H, ), so tested operations will broadcast
#   along the last dimension. The total FLOP is controlled by the feat_size no
#   matter how many heads are there.
# If num_heads = 0, it falls back to the normal element-wise operation without
#   broadcasting.
@utils.benchmark("flops", timeout=600)
@utils.parametrize("graph", ["ogbn-arxiv", "reddit", "ogbn-proteins"])
@utils.parametrize("feat_size", [4, 32, 256])
@utils.parametrize("num_heads", [0, 1, 4])
def track_flops(graph, feat_size, num_heads):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph, format="coo").to(device)
    if num_heads == 0:
        x = torch.randn(graph.num_nodes(), feat_size, device=device)
    else:
        x = torch.randn(
            graph.num_nodes(), num_heads, feat_size // num_heads, device=device
        )

    # dry run
    for i in range(3):
        y = dgl.ops.u_dot_v(graph, x, x)

    # timing
    with utils.Timer(device) as t:
        for i in range(10):
            y = dgl.ops.u_dot_v(graph, x, x)

    return calc_gflops(graph, feat_size, num_heads, t.elapsed_secs / 10)
