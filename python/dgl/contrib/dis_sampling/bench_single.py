# Benchmark sender
import mxnet as mx
import numpy as np
import random
import time

import sender

_SEED_SIZE = 8000
_VERTEX_NUM = 5000000
_NEIGHBOR_NUM = 16
_MAX_VERTICES = 1000000

_ITER = 1000

def load_big_graph():
    csr_graph = mx.nd.load('5_5_csr.nd')[0]
    return csr_graph

def create_seed(num_seed, max_id):
    seed = []
    for n in range(num_seed):
        seed.append(random.randint(0, max_id))
    seed_ndarray = mx.nd.array(seed, dtype='int64')
    return seed_ndarray

def start_single():
    graph = load_big_graph()
    seed = create_seed(_SEED_SIZE, _VERTEX_NUM)
    start = time.time()
    for n in range(_ITER):
        out = mx.nd.contrib.dgl_csr_neighbor_uniform_sample(
            graph, 
            seed, 
            num_args=2, 
            num_hops=1, 
            num_neighbor=_NEIGHBOR_NUM, 
            max_num_vertices=_MAX_VERTICES)

        str_ver_id = sender.serilize(out[0])
        str_layer = sender.serilize(out[2])
        str_csr_data = sender.serilize(out[1].data)
        str_csr_indices = sender.serilize(out[1].indices)
        str_csr_indptr = sender.serilize(out[1].indptr)

        ver_id = sender.deserilize(str_ver_id, np.int64)
        layer = sender.deserilize(str_layer, np.int64)
        csr_data = sender.deserilize(str_csr_data, np.int64)
        csr_indices = sender.deserilize(str_csr_indices, np.int64)
        csr_indptr = sender.deserilize(str_csr_indptr, np.int64)

    elapsed = (time.time() - start)    
    print("time: " + str(elapsed))

if __name__ == '__main__':
    start_single()