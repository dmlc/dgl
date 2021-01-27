import time
import dgl
import torch
import numpy as np

from .. import utils


@utils.benchmark('time', timeout=1200)
@utils.parametrize_cpu('graph_name', ['cora', 'livejournal', 'friendster'])
@utils.parametrize_gpu('graph_name', ['cora', 'livejournal'])
@utils.parametrize('format', ['coo', 'csc', 'csr'])
def track_time(graph_name, format):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format)
    graph = graph.to(device)
    graph = graph.formats([format])
    # dry run
    dgl.reverse(graph)

    # timing
    t0 = time.time()
    for i in range(10):
        gg = dgl.reverse(graph)
    t1 = time.time()

    return (t1 - t0) / 10
