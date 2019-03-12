from dgl import backend as F
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
    recv = dgl.contrib.sampling.SamplerReceiver(ip="127.0.0.1", port=50051)
    nodeflow = recv.Receive(g)
    print("----- trainer ------")
    print(nodeflow._node_mapping.todgltensor())
    print(nodeflow._edge_mapping.todgltensor())
    print(nodeflow._layer_offsets)
    print(nodeflow._block_offsets)
    print("----- trainer ------")

    time.sleep(3)  # wait all senders to finalize their jobs

def start_sampler():
    g = generate_rand_graph(100)
    sender = dgl.contrib.sampling.SamplerSender(ip="127.0.0.1", port=50051)
    for i, subg in enumerate(dgl.contrib.sampling.NeighborSampler(
            g, 1, 100, neighbor_type='in', num_workers=4)):
        print("----- sampler ------")
        print(subg._node_mapping.todgltensor())
        print(subg._edge_mapping.todgltensor())
        print(subg._layer_offsets)
        print(subg._block_offsets)
        print("----- sampler ------")
        sender.Send(subg)
        break

    time.sleep(1)

if __name__ == '__main__':
    pid = os.fork()
    if pid == 0:
        start_trainer()
    else:
        time.sleep(1)
        start_sampler()
