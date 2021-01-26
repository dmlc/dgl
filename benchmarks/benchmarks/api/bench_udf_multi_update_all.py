
import time
import dgl
import torch
import numpy as np
import dgl.function as fn


from .. import utils


@utils.benchmark('time', timeout=600)
@utils.parametrize('feat_size', [32, 128, 512])
@utils.parametrize('num_relations', [3, 6, 12])
@utils.thread_wrapped_func
def track_time(feat_size, num_relations):
    device = utils.get_bench_device()
    dd = {}
    candidate_edges = [dgl.data.CoraGraphDataset(verbose=False)[0].edges(), dgl.data.PubmedGraphDataset(verbose=False)[
        0].edges(), dgl.data.CiteseerGraphDataset(verbose=False)[0].edges()]
    for i in range(num_relations):
        dd[('n1', 'e_{}'.format(i), 'n2')] = candidate_edges[i %
                                                             len(candidate_edges)]
    graph = dgl.heterograph(dd)

    # Remove format strict
    graph = graph.to(device)
    graph.nodes['n1'].data['h'] = torch.randn(
        (graph.num_nodes('n1'), feat_size), device=device)
    graph.nodes['n2'].data['h'] = torch.randn(
        (graph.num_nodes('n2'), feat_size), device=device)

    # dry run
    update_dict = {}
    for i in range(num_relations):
        update_dict['e_{}'.format(i)] = (
            lambda edges: {'x': edges.src['h']}, lambda nodes: {'h_new': torch.sum(nodes.mailbox['x'], dim=1)})
    graph.multi_update_all(
        update_dict,
        "sum")

    # timing
    t0 = time.time()
    for i in range(3):
        graph.multi_update_all(
            update_dict,
            "sum")
    t1 = time.time()

    return (t1 - t0) / 3
