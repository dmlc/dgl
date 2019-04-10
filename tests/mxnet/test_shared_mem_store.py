import dgl
import time
import numpy as np
from multiprocessing import Process
from scipy import sparse as spsp
import mxnet as mx
import backend as F
import unittest

num_nodes = 100
num_edges = int(num_nodes * num_nodes * 0.1)

def worker_func(worker_id):
    time.sleep(3)
    print("worker starts")
    np.random.seed(0)
    csr = (spsp.random(num_nodes, num_nodes, density=0.1, format='csr') != 0).astype(np.int64)

    g = dgl.contrib.graph_store.create_graph_from_store("test_graph5", "shared_mem")
    # Verify the graph structure loaded from the shared memory.
    src, dst = g.all_edges()
    coo = csr.tocoo()
    assert F.array_equal(dst, F.tensor(coo.row))
    assert F.array_equal(src, F.tensor(coo.col))
    assert F.array_equal(g.ndata['feat'][0], F.tensor(np.arange(10), dtype=np.float32))
    assert F.array_equal(g.edata['feat'][0], F.tensor(np.arange(10), dtype=np.float32))
    g.ndata['test4'] = mx.nd.zeros((g.number_of_nodes(), 10))
    g.edata['test4'] = mx.nd.zeros((g.number_of_edges(), 10))
    if worker_id == 0:
        time.sleep(3)
        print(g.worker_id)
        g.ndata['test4'][0] = 1
        g.edata['test4'][0] = 2
    else:
        time.sleep(5)
        print(g.worker_id)
        assert np.all(g.ndata['test4'][0].asnumpy() == 1)
        assert np.all(g.edata['test4'][0].asnumpy() == 2)
    g.destroy()

def server_func(num_workers):
    print("server starts")
    np.random.seed(0)
    csr = (spsp.random(num_nodes, num_nodes, density=0.1, format='csr') != 0).astype(np.int64)

    g = dgl.contrib.graph_store.create_graph_store_server(csr, "test_graph5", "shared_mem", num_workers,
                                                          False, edge_dir="in")
    assert num_nodes == g._graph.number_of_nodes()
    assert num_edges == g._graph.number_of_edges()
    g.ndata['feat'] = mx.nd.arange(num_nodes * 10).reshape((num_nodes, 10))
    g.edata['feat'] = mx.nd.arange(num_edges * 10).reshape((num_edges, 10))
    g.run()

@unittest.skip("disable shared memory test temporarily")
def test_worker_server():
    serv_p = Process(target=server_func, args=(2,))
    work_p1 = Process(target=worker_func, args=(0,))
    work_p2 = Process(target=worker_func, args=(1,))
    serv_p.start()
    work_p1.start()
    work_p2.start()
    serv_p.join()
    work_p1.join()
    work_p2.join()

if __name__ == '__main__':
    test_worker_server()
