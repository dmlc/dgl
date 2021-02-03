import time
import dgl
import torch
import numpy as np
import dgl.function as fn

from .. import utils


@utils.benchmark('time', timeout=7200)
@utils.parametrize('graph_name', ['cora', 'livejournal'])
@utils.parametrize('format', ['coo', 'csr'])
@utils.parametrize_cpu('feat_size', [8, 128, 512])
@utils.parametrize_gpu('feat_size', [8, 32, 256])
@utils.parametrize('msg_type', ['copy_u', 'u_mul_e'])
@utils.parametrize('reduce_type', ['sum', 'mean', 'max'])
def track_time(graph_name, format, feat_size, msg_type, reduce_type):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format)
    graph = graph.to(device)
    graph.ndata['h'] = torch.randn(
        (graph.num_nodes(), feat_size), device=device)
    graph.edata['e'] = torch.randn(
        (graph.num_edges(), 1), device=device)
    
    msg_builtin_dict = {
        'copy_u': fn.copy_u('h', 'x'),
        'u_mul_e': fn.u_mul_e('h', 'e','x'),
    }

    reduce_builtin_dict = {
        'sum': fn.sum('x', 'h_new'),
        'mean': fn.mean('x', 'h_new'),
        'max': fn.max('x', 'h_new'),
    }

    # dry run
    graph.update_all(msg_builtin_dict[msg_type], reduce_builtin_dict[reduce_type])

    # timing
   
    with utils.Timer() as t:
        for i in range(3):
            graph.update_all(msg_builtin_dict[msg_type], reduce_builtin_dict[reduce_type])

    return t.elapsed_secs / 3

