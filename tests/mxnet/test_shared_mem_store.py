import dgl
import time
import numpy as np
from multiprocessing import Process
from scipy import sparse as spsp

graph = (spsp.random(100, 100, density=0.1, format='coo') != 0).astype(np.int64)

def worker_func():
    print("worker starts")
    time.sleep(10)
    print("worker create graph store")
    store = dgl.contrib.graph_store.create_graph_store_client(graph, "test_graph", "shared_mem", False)

def server_func():
    print("server starts")
    dgl.contrib.graph_store.run_graph_store_server(graph, "test_graph", "shared_mem", False, 1)

serv_p = Process(target=server_func, args=())
#work_p = Process(target=worker_func, args=())
serv_p.start()
#work_p.start()
serv_p.join()
#work_p.join()
