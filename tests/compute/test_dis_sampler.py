import backend as F
import numpy as np
import scipy as sp
import dgl
from dgl import utils

import os
import time

def generate_rand_graph(n):
    arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
    return dgl.DGLGraph(arr, readonly=True)

def start_trainer():
    g = generate_rand_graph(100)
    sampler = dgl.contrib.sampling.SamplerReceiver(graph=g, addr='127.0.0.1:50051', num_sender=1)
    for subg in sampler:
        seed_ids = subg.layer_parent_nid(-1)
        assert len(seed_ids) == 1
        src, dst, eid = g.in_edges(seed_ids, form='all')
        assert subg.number_of_nodes() == len(src) + 1
        assert subg.number_of_edges() == len(src)

        assert seed_ids == subg.layer_parent_nid(-1)
        child_src, child_dst, child_eid = subg.in_edges(subg.layer_nid(-1), form='all')
        assert F.array_equal(child_src, subg.layer_nid(0))

        src1 = subg.map_to_parent_nid(child_src)
        assert F.array_equal(src1, src)

def start_sampler():
    g = generate_rand_graph(100)
    namebook = { 0:'127.0.0.1:50051' }
    sender = dgl.contrib.sampling.SamplerSender(namebook)
    for i, subg in enumerate(dgl.contrib.sampling.NeighborSampler(
            g, 1, 100, neighbor_type='in', num_workers=4)):
        sender.send(subg, 0)
    sender.signal(0)

if __name__ == '__main__':
    pid = os.fork()
    if pid == 0:
        start_trainer()
    else:
        time.sleep(2) # wait trainer start
        start_sampler()
