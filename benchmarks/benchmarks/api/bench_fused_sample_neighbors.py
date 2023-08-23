import time

import dgl
import dgl.function as fn

import numpy as np
import torch

from .. import utils


@utils.benchmark("time")
@utils.parametrize_cpu("graph_name", ["livejournal", "reddit"])
@utils.parametrize_gpu("graph_name", ["ogbn-arxiv", "reddit"])
@utils.parametrize("format", ["csr", "csc"])
@utils.parametrize("seed_nodes_num", [200, 5000, 20000])
@utils.parametrize("fanout", [5, 20, 40])
def track_time(graph_name, format, seed_nodes_num, fanout):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format).to(device)

    edge_dir = "in" if format == "csc" else "out"
    seed_nodes = np.random.randint(0, graph.num_nodes(), seed_nodes_num)
    seed_nodes = torch.from_numpy(seed_nodes).to(device)

    # dry run
    for i in range(3):
        dgl.sampling.sample_neighbors_fused(
            graph, seed_nodes, fanout, edge_dir=edge_dir
        )

    # timing
    with utils.Timer() as t:
        for i in range(50):
            dgl.sampling.sample_neighbors_fused(
                graph, seed_nodes, fanout, edge_dir=edge_dir
            )

    return t.elapsed_secs / 50
