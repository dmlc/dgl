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
@utils.parametrize('msg_type', ['copy_u', 'copy_v', 'u_mul_e'])
@utils.parametrize('reduce_type', ['sum', 'mean', 'max'])
def track_time(graph_name, format, feat_size, msg_type, reduce_type):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format)
    # Remove format strict
    graph = graph.formats(['coo', 'csr', 'csc'])
    graph = graph.to(device)
    graph.ndata['h'] = torch.randn(
        (graph.num_nodes(), feat_size), device=device)
    graph.edata['e'] = torch.randn(
        (graph.num_edges(), feat_size), device=device)
    
    msg_udf_dict = {
        'copy_u': lambda edges: {'x': edges.src['h']},
        'copy_v': lambda edges: {'x': edges.dst['h']},
        'u_mul_e': lambda edges: {'x': edges.src['h']*edges.data['e']},
    }

    reduct_udf_dict = {
        'sum': lambda nodes: {'h_new': torch.sum(nodes.mailbox['x'], dim=1)},
        'mean': lambda nodes: {'h_new': torch.mean(nodes.mailbox['x'], dim=1)},
        'max': lambda nodes: {'h_new': torch.max(nodes.mailbox['x'], dim=1)[0]},
    }

    # dry run
    graph.update_all(msg_udf_dict[msg_type], reduct_udf_dict[reduce_type])

    # timing
    t0 = time.time()
    for i in range(3):
        graph.update_all(msg_udf_dict[msg_type], reduct_udf_dict[reduce_type])
    t1 = time.time()

    return (t1 - t0) / 3

