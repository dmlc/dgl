# This is a demo code shows that how to implement a remote sender
import mxnet as mx
import numpy as np

import sender

_REMOTE_ADDR = 'localhost:50051'

def create_graph():
    """ 
        Create a small demo graph:

        [[ 0  1  2  3  4]
         [ 5  0  6  7  8]
         [ 9 10  0 11 12]
         [13 14 15  0 16]
         [17 18 19 20  0]]
    """
    shape = (5, 5)
    list = []
    for i in range(20):
        list.append(i+1)
    data_np = np.array(list, dtype=np.int64)

    list = []
    for i in range(5):
        for j in range(5):
            if j != i:
                list.append(j);
    indices_np = np.array(list, dtype=np.int64)

    list = []
    for i in range(6):
        list.append(i*4)
    indptr_np = np.array(list, dtype=np.int64)

    csr_graph = mx.nd.sparse.csr_matrix((data_np, indices_np, indptr_np), shape=shape)
    return csr_graph

def start_sender():
    my_sender = sender.Sender(addr=_REMOTE_ADDR) 
    func = mx.nd.contrib.dgl_csr_neighbor_uniform_sample
    graph = create_graph()
    seeds = mx.nd.array([0,1,2,3,4], dtype=np.int64)
    num_args = 2
    num_hops = 1
    num_neighbor = 2
    max_num_vertices= 5
    my_sender.bind(func, graph, seeds, num_args, num_hops, num_neighbor, max_num_vertices)
    my_sender.start()

if __name__ == '__main__':
    start_sender()