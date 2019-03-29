import dgl
import time
import numpy as np
from multiprocessing import Process
from scipy import sparse as spsp
import mxnet as mx

graph = (spsp.random(100, 100, density=0.1, format='coo') != 0).astype(np.int64)

def worker_func(worker_id):
    print("worker starts")
    store = dgl.contrib.graph_store.create_graph_store_client(graph, "test_graph", "shared_mem", False)
    store.init_ndata('test4', 10, 0)
    if worker_id == 0:
        store.ndata['test4'][0] = 1
        print('set ndata')
        print(store.ndata['test4'][0].asnumpy())
    else:
        time.sleep(1)
        print(store.ndata['test4'][0].asnumpy())
        assert np.all(store.ndata['test4'][0].asnumpy() == 1)

def server_func():
    print("server starts")
    n = 100
    csr = (sp.sparse.random(n, n, density=0.1, format='csr') != 0).astype(np.int64)
    idx = dgl.graph_index.GraphIndex(multigraph=False, readonly=True)
    idx.from_scipy_csr_matrix(csr, 'out', '/test_graph_struct')
    assert idx.number_of_nodes() == n
    assert idx.number_of_edges() == csr.nnz
    src, dst, eid = idx.edges()
    src, dst, eid = src.tousertensor(), dst.tousertensor(), eid.tousertensor()
    coo = csr.tocoo()
    assert F.array_equal(src, F.tensor(coo.row))
    assert F.array_equal(dst, F.tensor(coo.col))

    dgl.contrib.graph_store.run_graph_store_server(graph, "test_graph", "shared_mem", False, 1)

serv_p = Process(target=server_func, args=())
work_p1 = Process(target=worker_func, args=(0,))
work_p2 = Process(target=worker_func, args=(1,))
serv_p.start()
work_p1.start()
work_p2.start()
serv_p.join()
work_p1.join()
work_p2.join()
