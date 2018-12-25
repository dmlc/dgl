# This is a demo code shows that how to 
# implement a remote sampler node
import mxnet as mx
import numpy as np

import time
import sampler

_REMOTE_ADDR = 'localhost:50051'

def create_graph():
    """ Load huge graph from checkpoint
        or generate a small graph for test.

        Demo small graph:

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


class MySampler(sampler.Sampler):
    """ User-defined sampler
    """

    def Sample(self):
        """ Sample sub-graph from huge graph
            For now, it is just a small demo, which takes
            seed = [0, 1, 2, 3, 4] and uses uniform sampling API
        """
        time.sleep(1)
        out = mx.nd.contrib.dgl_csr_neighbor_uniform_sample(
            self.Graph,
            self.Seeds,
            num_args=self.NumArgs,
            num_hops=self.NumHops,
            num_neighbor=self.NumNeighbor,
            max_num_vertices=self.MaxVertices)

        print("Sample finish ...");

        return out, self.MaxVertices

# Connect to server node and sample sub-graphs in a loop
def start_sample():
    graph = create_graph()
    print("--------Graph---------")
    print(graph.asnumpy())
    print("Remote address: ", _REMOTE_ADDR)
    sampler = MySampler(graph, _REMOTE_ADDR)
    sampler.Start()

if __name__ == '__main__':
    start_sample()