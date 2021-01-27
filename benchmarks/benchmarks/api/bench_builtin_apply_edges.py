import time
import dgl
import torch
import numpy as np
import dgl.function as fn

from .. import utils


@utils.benchmark('time', timeout=600)
@utils.parametrize('graph_name', ['cora', 'livejournal'])
@utils.parametrize('format', ['coo', 'csr'])
@utils.parametrize('feat_size', [8, 32, 128, 512])
@utils.parametrize('reduce_type', ['u->e', 'u+v'])
def track_time(graph_name, format, feat_size, reduce_type):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format)
    graph = graph.to(device)
    graph.ndata['h'] = torch.randn(
        (graph.num_nodes(), feat_size), device=device)

    reduce_builtin_dict = {
        'u->e': fn.copy_u('h', 'x'),
        'u+v': fn.u_add_v('h', 'h', 'x'),
    }

    # dry run
    graph.apply_edges(reduce_builtin_dict[reduce_type])

    # timing
    t0 = time.time()
    for i in range(3):
        graph.apply_edges(reduce_builtin_dict[reduce_type])
    t1 = time.time()

    return (t1 - t0) / 3
