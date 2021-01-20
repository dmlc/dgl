import time
import dgl
import torch
import numpy as np

from .. import utils


@utils.skip_if_gpu()
@utils.benchmark('time', timeout=1200)
@utils.parametrize('graph_name', ['reddit'])
@utils.parametrize('k', [2, 4, 8])
def track_time(graph_name, k):
    device = utils.get_bench_device()
    data = utils.process_data(graph_name)
    graph = data[0]
    # dry run
    dry_run_data = utils.process_data('pubmed')
    gg = dgl.transform.metis_partition(dry_run_data[0], k)

    # timing
    t0 = time.time()
    for i in range(3):
        gg = dgl.transform.metis_partition(graph, k)
    t1 = time.time()

    return (t1 - t0) / 3 
