import dgl
import sys
import random
import time
import numpy as np
from multiprocessing import Process, Manager
from scipy import sparse as spsp
import mxnet as mx
import backend as F
import unittest
import dgl.function as fn
import traceback
from numpy.testing import assert_almost_equal

num_nodes = 100
num_edges = int(num_nodes * num_nodes * 0.1)
rand_port = random.randint(5000, 8000)
print('run graph store with port ' + str(rand_port), file=sys.stderr)

def check_array_shared_memory(g, worker_id, arrays):
    if worker_id == 0:
        for i, arr in enumerate(arrays):
            arr[0] = i
        g._sync_barrier(60)
    else:
        g._sync_barrier(60)
        for i, arr in enumerate(arrays):
            assert_almost_equal(arr[0].asnumpy(), i)

def create_graph_store(graph_name):
    for _ in range(10):
        try:
            g = dgl.contrib.graph_store.create_graph_from_store(graph_name, "shared_mem",
                                                                port=rand_port)
            return g
        except ConnectionError as e:
            traceback.print_exc()
            time.sleep(1)
    return None

def check_init_func(worker_id, graph_name, return_dict):
    time.sleep(3)
    print("worker starts")
    np.random.seed(0)
    csr = (spsp.random(num_nodes, num_nodes, density=0.1, format='csr') != 0).astype(np.int64)

    # Verify the graph structure loaded from the shared memory.
    try:
        g = create_graph_store(graph_name)
        if g is None:
            return_dict[worker_id] = -1
            return

        src, dst = g.all_edges()
        coo = csr.tocoo()
        assert F.array_equal(dst, F.tensor(coo.row))
        assert F.array_equal(src, F.tensor(coo.col))
        assert F.array_equal(g.nodes[0].data['feat'], F.tensor(np.arange(10), dtype=np.float32))
        assert F.array_equal(g.edges[0].data['feat'], F.tensor(np.arange(10), dtype=np.float32))
        g.init_ndata('test4', (g.number_of_nodes(), 10), 'float32')
        g.init_edata('test4', (g.number_of_edges(), 10), 'float32')
        g._sync_barrier(60)
        check_array_shared_memory(g, worker_id, [g.nodes[:].data['test4'], g.edges[:].data['test4']])

        data = g.nodes[:].data['test4']
        g.set_n_repr({'test4': mx.nd.ones((1, 10)) * 10}, u=[0])
        assert_almost_equal(data[0].asnumpy(), np.squeeze(g.nodes[0].data['test4'].asnumpy()))

        data = g.edges[:].data['test4']
        g.set_e_repr({'test4': mx.nd.ones((1, 10)) * 20}, edges=[0])
        assert_almost_equal(data[0].asnumpy(), np.squeeze(g.edges[0].data['test4'].asnumpy()))

        g.destroy()
        return_dict[worker_id] = 0
    except Exception as e:
        return_dict[worker_id] = -1
        g.destroy()
        print(e, file=sys.stderr)
        traceback.print_exc()

def server_func(num_workers, graph_name):
    print("server starts")
    np.random.seed(0)
    csr = (spsp.random(num_nodes, num_nodes, density=0.1, format='csr') != 0).astype(np.int64)

    g = dgl.contrib.graph_store.create_graph_store_server(csr, graph_name, "shared_mem", num_workers,
                                                          False, edge_dir="in", port=rand_port)
    assert num_nodes == g._graph.number_of_nodes()
    assert num_edges == g._graph.number_of_edges()
    g.ndata['feat'] = mx.nd.arange(num_nodes * 10).reshape((num_nodes, 10))
    g.edata['feat'] = mx.nd.arange(num_edges * 10).reshape((num_edges, 10))
    g.run()

def test_init():
    manager = Manager()
    return_dict = manager.dict()
    serv_p = Process(target=server_func, args=(2, 'test_graph1'))
    work_p1 = Process(target=check_init_func, args=(0, 'test_graph1', return_dict))
    work_p2 = Process(target=check_init_func, args=(1, 'test_graph1', return_dict))
    serv_p.start()
    work_p1.start()
    work_p2.start()
    serv_p.join()
    work_p1.join()
    work_p2.join()
    for worker_id in return_dict.keys():
        assert return_dict[worker_id] == 0, "worker %d fails" % worker_id


def check_compute_func(worker_id, graph_name, return_dict):
    time.sleep(3)
    print("worker starts")
    try:
        g = create_graph_store(graph_name)
        if g is None:
            return_dict[worker_id] = -1
            return

        g._sync_barrier(60)
        in_feats = g.nodes[0].data['feat'].shape[1]

        # Test update all.
        g.update_all(fn.copy_src(src='feat', out='m'), fn.sum(msg='m', out='preprocess'))
        adj = g.adjacency_matrix()
        tmp = mx.nd.dot(adj, g.nodes[:].data['feat'])
        assert_almost_equal(g.nodes[:].data['preprocess'].asnumpy(), tmp.asnumpy())
        g._sync_barrier(60)
        check_array_shared_memory(g, worker_id, [g.nodes[:].data['preprocess']])

        # Test apply nodes.
        data = g.nodes[:].data['feat']
        g.apply_nodes(func=lambda nodes: {'feat': mx.nd.ones((1, in_feats)) * 10}, v=0)
        assert_almost_equal(data[0].asnumpy(), np.squeeze(g.nodes[0].data['feat'].asnumpy()))

        # Test apply edges.
        data = g.edges[:].data['feat']
        g.apply_edges(func=lambda edges: {'feat': mx.nd.ones((1, in_feats)) * 10}, edges=0)
        assert_almost_equal(data[0].asnumpy(), np.squeeze(g.edges[0].data['feat'].asnumpy()))

        g.init_ndata('tmp', (g.number_of_nodes(), 10), 'float32')
        data = g.nodes[:].data['tmp']
        # Test pull
        g.pull(1, fn.copy_src(src='feat', out='m'), fn.sum(msg='m', out='tmp'))
        assert_almost_equal(data[1].asnumpy(), np.squeeze(g.nodes[1].data['preprocess'].asnumpy()))

        # Test send_and_recv
        in_edges = g.in_edges(v=2)
        g.send_and_recv(in_edges, fn.copy_src(src='feat', out='m'), fn.sum(msg='m', out='tmp'))
        assert_almost_equal(data[2].asnumpy(), np.squeeze(g.nodes[2].data['preprocess'].asnumpy()))

        g.destroy()
        return_dict[worker_id] = 0
    except Exception as e:
        return_dict[worker_id] = -1
        g.destroy()
        print(e, file=sys.stderr)
        traceback.print_exc()

def test_compute():
    manager = Manager()
    return_dict = manager.dict()
    serv_p = Process(target=server_func, args=(2, 'test_graph3'))
    work_p1 = Process(target=check_compute_func, args=(0, 'test_graph3', return_dict))
    work_p2 = Process(target=check_compute_func, args=(1, 'test_graph3', return_dict))
    serv_p.start()
    work_p1.start()
    work_p2.start()
    serv_p.join()
    work_p1.join()
    work_p2.join()
    for worker_id in return_dict.keys():
        assert return_dict[worker_id] == 0, "worker %d fails" % worker_id

def check_sync_barrier(graph_name, return_dict):
    time.sleep(3)
    print("worker starts")
    try:
        g = create_graph_store(graph_name)
        if g is None:
            return_dict[worker_id] = -1
            return

        start = time.time()
        g._sync_barrier(30)
        # this is very loose.
        assert abs(time.time() - start) < 5
        g.destroy()
        return_dict[worker_id] = 0
    except Exception as e:
        return_dict[worker_id] = -1
        g.destroy()
        print(e, file=sys.stderr)
        traceback.print_exc()


def test_sync_barrier():
    manager = Manager()
    return_dict = manager.dict()
    serv_p = Process(target=server_func, args=(2, 'test_graph4'))
    work_p1 = Process(target=check_sync_barrier, args=('test_graph4', return_dict))
    serv_p.start()
    work_p1.start()
    serv_p.join()
    work_p1.join()
    for worker_id in return_dict.keys():
        assert return_dict[worker_id] == 0, "worker %d fails" % worker_id

if __name__ == '__main__':
    test_init()
    test_sync_barrier()
    test_compute()
