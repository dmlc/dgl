
import time
import dgl
import torch
import numpy as np
import dgl.function as fn


from .. import utils


@utils.benchmark('time')
@utils.parametrize('num_relations', [5, 50, 500])
def track_time(num_relations):
    dd = {}
    candidate_edges = [dgl.data.CoraGraphDataset(verbose=False)[0].edges(), dgl.data.PubmedGraphDataset(verbose=False)[
        0].edges(), dgl.data.CiteseerGraphDataset(verbose=False)[0].edges()]
    for i in range(num_relations):
        dd[('n1', 'e_{}'.format(i), 'n2')] = candidate_edges[i %
                                                             len(candidate_edges)]

    # dry run
    graph = dgl.heterograph(dd)

    # timing
    t0 = time.time()
    for i in range(3):
        graph = dgl.heterograph(dd)
    t1 = time.time()

    return (t1 - t0) / 3
