import time
import dgl
import torch

from .. import utils

@utils.benchmark('time')
@utils.parametrize('graph', ['reddit'])
@utils.parametrize('feat_size', [4, 32, 256])
def track_time(graph, feat_size):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph, format='csc').to(device)
    x = torch.randn(graph.num_nodes(), feat_size, device=device)

    # dry run
    for i in range(3):
        y = dgl.ops.copy_u_sum(graph, x)

    # timing
    accum = 0.
    for i in range(10):
        with utils.TorchOpTimer(device) as timer:
            y = dgl.ops.copy_u_sum(graph, x)
        accum += timer.time

    return accum / 10
