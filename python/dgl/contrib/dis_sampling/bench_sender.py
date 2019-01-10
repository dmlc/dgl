# Benchmark sender
import mxnet as mx
import numpy as np
import random

import sender

_SEED_SIZE = 8000
_VERTEX_NUM = 5000000
_NEIGHBOR_NUM = 16
_MAX_VERTICES = 100000

_REMOTE_ADDR = '172.31.65.30:2049'

def load_big_graph():
    csr_graph = mx.nd.load('5_5_csr.nd')[0]
    return csr_graph

def create_seed(num_seed, max_id):
    seed = []
    for n in range(num_seed):
        seed.append(random.randint(0, max_id))
    seed_ndarray = mx.nd.array(seed, dtype='int64')
    return seed_ndarray

def start_sender():
    graph = load_big_graph()
    seed = create_seed(_SEED_SIZE, _VERTEX_NUM)
    func = mx.nd.contrib.dgl_csr_neighbor_uniform_sample
    my_sender = sender.Sender(addr=_REMOTE_ADDR) 
    my_sender.bind(func, graph, seed, 2, 1, _NEIGHBOR_NUM, _MAX_VERTICES)
    my_sender.start()

if __name__ == '__main__':
    start_sender()