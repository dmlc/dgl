import time
import dgl
import torch
import numpy as np
import dgl.function as fn

from .. import utils


@utils.benchmark('time', timeout=7200)
@utils.parametrize('graph_name', ['cora', 'livejournal'])
@utils.parametrize('format', ['coo', 'csr'])
@utils.parametrize('feat_size', [8, 32, 128, 512])
@utils.parametrize('reduce_type', ['u->e', 'v->e', 'u+v', 'u*v', 'udotv'])
def track_time(graph_name, format, feat_size, reduce_type):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format)
    # Remove format strict
    graph = graph.formats(['coo', 'csr', 'csc'])
    graph = graph.to(device)
    graph.ndata['h'] = torch.randn(
        (graph.num_nodes(), feat_size), device=device)

    reduce_udf_dict = {
        'u->e': lambda edges: {'x': edges.src['h']},
        'v->e': lambda edges: {'x': edges.dst['h']},
        'u+v': lambda edges: {'x': edges.src['h']+edges.dst['h']},
        'u*v': lambda edges: {'x': edges.src['h']*edges.dst['h']},
        'udotv': lambda edges: {'x': (edges.src['h']*edges.dst['h']).sum(-1)},
    }

    # dry run
    graph.apply_edges(reduce_udf_dict[reduce_type])

    # timing
    t0 = time.time()
    for i in range(3):
        graph.apply_edges(reduce_udf_dict[reduce_type])
    t1 = time.time()

    return (t1 - t0) / 3
