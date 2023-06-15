import time

import dgl

import numpy as np
import torch

from .. import utils


@utils.benchmark("time", timeout=600)
@utils.parametrize_cpu(
    "graph_name", ["cora", "pubmed", "ogbn-arxiv", "livejournal", "friendster"]
)
@utils.parametrize_gpu("graph_name", ["cora", "livejournal"])
@utils.parametrize(
    "format",
    [
        ("coo", "csc"),
        ("csc", "coo"),
        ("coo", "csr"),
        ("csr", "coo"),
        ("csr", "csc"),
        ("csc", "csr"),
    ],
)
def track_time(graph_name, format):
    from_format, to_format = format
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, from_format)
    graph = graph.to(device)
    if format == ("coo", "csr") and graph_name == "friendster":
        # Mark graph as sorted to check performance for COO matrix marked as
        # sorted. Note that friendster dataset is already sorted.
        graph = dgl.graph(graph.edges(), row_sorted=True)
    graph = graph.formats([from_format])
    # dry run
    graph.formats([to_format])

    # timing
    with utils.Timer() as t:
        for i in range(10):
            gg = graph.formats([to_format])

    return t.elapsed_secs / 10
