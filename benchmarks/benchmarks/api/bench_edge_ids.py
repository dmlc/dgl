import time

import dgl

import numpy as np
import torch

from .. import utils


# edge_ids is not supported on cuda
# @utils.skip_if_gpu()
@utils.benchmark("time", timeout=1200)
@utils.parametrize_cpu("graph_name", ["cora", "livejournal", "friendster"])
@utils.parametrize_gpu("graph_name", ["cora", "livejournal"])
@utils.parametrize("format", ["coo", "csr", "csc"])
@utils.parametrize("fraction", [0.01, 0.1])
@utils.parametrize("return_uv", [True, False])
def track_time(graph_name, format, fraction, return_uv):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format)
    coo_graph = utils.get_graph(graph_name, "coo")
    graph = graph.to(device)
    eids = np.random.choice(
        np.arange(graph.num_edges(), dtype=np.int64),
        int(graph.num_edges() * fraction),
    )
    eids = torch.tensor(eids, device="cpu", dtype=torch.int64)
    u, v = coo_graph.find_edges(eids)
    del coo_graph, eids
    u = u.to(device)
    v = v.to(device)
    # dry run
    for i in range(10):
        out = graph.edge_ids(u[0], v[0])

    # timing

    with utils.Timer() as t:
        for i in range(3):
            edges = graph.edge_ids(u, v, return_uv=return_uv)

    return t.elapsed_secs / 3
