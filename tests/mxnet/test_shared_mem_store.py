import dgl
import time
import numpy as np
from multiprocessing import Process
from scipy import sparse as spsp
import mxnet as mx
import backend as F

num_nodes = 100
num_edges = int(num_nodes * num_nodes * 0.1)

def worker_func(worker_id):
    print("worker starts")
    time.sleep(1)
    np.random.seed(0)
    csr = (spsp.random(num_nodes, num_nodes, density=0.1, format='csr') != 0).astype(np.int64)
    graph = dgl.graph_index.GraphIndex(multigraph=False, readonly=True)
    graph.from_shared_mem_csr_matrix('/test_graph_struct', num_nodes, num_edges, 'in')

    # Verify the graph structure loaded from the shared memory.
    src, dst, eid = graph.edges()
    src, dst, eid = src.tousertensor(), dst.tousertensor(), eid.tousertensor()
    coo = csr.tocoo()
    assert F.array_equal(dst, F.tensor(coo.row))
    assert F.array_equal(src, F.tensor(coo.col))

    store = dgl.contrib.graph_store.create_graph_store_client(graph, "test_graph", "shared_mem", False)
    store.init_ndata('test4', 10, 0)
    if worker_id == 0:
        store.ndata['test4'][0] = 1
    else:
        time.sleep(1)
        assert np.all(store.ndata['test4'][0].asnumpy() == 1)
    store.destroy()

def server_func(num_workers):
    print("server starts")
    np.random.seed(0)
    csr = (spsp.random(num_nodes, num_nodes, density=0.1, format='csr') != 0).astype(np.int64)
    graph = dgl.graph_index.GraphIndex(multigraph=False, readonly=True)
    graph.from_csr_matrix(csr.indptr, csr.indices, 'in', '/test_graph_struct')
    assert graph.number_of_nodes() == num_nodes
    assert graph.number_of_edges() == csr.nnz

    dgl.contrib.graph_store.run_graph_store_server(graph, "test_graph", "shared_mem", False, num_workers)

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
