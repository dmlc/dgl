import dgl
import time
import numpy as np
from multiprocessing import Process
from scipy import sparse as spsp

graph = (spsp.random(100, 100, density=0.1, format='coo') != 0).astype(np.int64)

def worker_func(worker_id):
    print("worker starts")
    store = dgl.contrib.graph_store.create_graph_store_client(graph, "test_graph", "shared_mem", False)
    store.init_ndata('test4', 10, 0)

def server_func():
    print("server starts")
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
