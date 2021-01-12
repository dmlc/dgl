import time
import dgl
import torch
import numpy as np

from .. import utils

# edge_ids is not supported on cuda
@utils.skip_if_gpu()
@utils.benchmark('time', timeout=1200)
@utils.parametrize_cpu('graph_name', ['cora', 'livejournal', 'friendster'])
@utils.parametrize_gpu('graph_name', ['cora', 'livejournal'])
@utils.parametrize('format', ['csc'])  # csc is not supported
@utils.parametrize('fraction', [0.01, 0.1])
def track_time(graph_name, format, fraction):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format)
    graph = graph.to(device)
    eids = np.random.RandomState(6666).choice(
        np.arange(graph.num_edges(), dtype=np.int64), int(graph.num_edges()*fraction))
    eids = torch.tensor(eids, device=device, dtype=torch.int64)
    u, v = graph.find_edges(eids)
    # dry run
    for i in range(10):
        out = graph.edge_ids(u[0], v[0])

    # timing
    t0 = time.time()
    for i in range(10):
        edges = graph.edge_ids(u, v)
    t1 = time.time()

    return (t1 - t0) / 10
