""" NOTE(zihao) The unittest on shared memory store is temporally disabled because we 
have not fixed the bug described in https://github.com/dmlc/dgl/issues/755 yet.
The bug causes CI failures occasionally but does not affect other parts of DGL.
As a result, we decide to disable this test until we fixed the bug.
"""
import dgl
import sys
import random
import time
import numpy as np
from numpy.testing import assert_array_equal
from multiprocessing import Process, Manager, Condition, Value
from scipy import sparse as spsp
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
            arr[0] = i + 10
        g._sync_barrier(60)
    else:
        g._sync_barrier(60)
        for i, arr in enumerate(arrays):
            assert_almost_equal(F.asnumpy(arr[0]), i + 10)

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
        assert_array_equal(F.asnumpy(dst), coo.row)
        assert_array_equal(F.asnumpy(src), coo.col)
        feat = F.asnumpy(g.nodes[0].data['feat'])
        assert_array_equal(np.squeeze(feat), np.arange(10, dtype=feat.dtype))
        feat = F.asnumpy(g.edges[0].data['feat'])
        assert_array_equal(np.squeeze(feat), np.arange(10, dtype=feat.dtype))
        g.init_ndata('test4', (g.number_of_nodes(), 10), 'float32')
        g.init_edata('test4', (g.number_of_edges(), 10), 'float32')
        g._sync_barrier(60)
        check_array_shared_memory(g, worker_id, [g.nodes[:].data['test4'], g.edges[:].data['test4']])
        g._sync_barrier(60)

        data = g.nodes[:].data['test4']
        g.set_n_repr({'test4': F.ones((1, 10)) * 10}, u=[0])
        assert_almost_equal(F.asnumpy(data[0]), np.squeeze(F.asnumpy(g.nodes[0].data['test4'])))

        data = g.edges[:].data['test4']
        g.set_e_repr({'test4': F.ones((1, 10)) * 20}, edges=[0])
        assert_almost_equal(F.asnumpy(data[0]), np.squeeze(F.asnumpy(g.edges[0].data['test4'])))

        g.destroy()
        return_dict[worker_id] = 0
    except Exception as e:
        return_dict[worker_id] = -1
        g.destroy()
        print(e, file=sys.stderr)
        traceback.print_exc()

def server_func(num_workers, graph_name, server_init):
    np.random.seed(0)
    csr = (spsp.random(num_nodes, num_nodes, density=0.1, format='csr') != 0).astype(np.int64)

    g = dgl.contrib.graph_store.create_graph_store_server(csr, graph_name, "shared_mem", num_workers,
                                                          False, edge_dir="in", port=rand_port)
    assert num_nodes == g._graph.number_of_nodes()
    assert num_edges == g._graph.number_of_edges()
    nfeat = np.arange(0, num_nodes * 10).astype('float32').reshape((num_nodes, 10))
    efeat = np.arange(0, num_edges * 10).astype('float32').reshape((num_edges, 10))
    g.ndata['feat'] = F.tensor(nfeat)
    g.edata['feat'] = F.tensor(efeat)
    server_init.value = 1
    g.run()

def test_init():
    manager = Manager()
    return_dict = manager.dict()

    # make server init before worker
    server_init = Value('i', False)
    serv_p = Process(target=server_func, args=(2, 'test_graph1', server_init))
    serv_p.start()
    while server_init.value == 0:
      time.sleep(1)
    work_p1 = Process(target=check_init_func, args=(0, 'test_graph1', return_dict))
    work_p2 = Process(target=check_init_func, args=(1, 'test_graph1', return_dict))
    work_p1.start()
    work_p2.start()
    serv_p.join()
    work_p1.join()
    work_p2.join()
    for worker_id in return_dict.keys():
        assert return_dict[worker_id] == 0, "worker %d fails" % worker_id

def check_compute_func(worker_id, graph_name, return_dict):
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
        tmp = F.spmm(adj, g.nodes[:].data['feat'])
        assert_almost_equal(F.asnumpy(g.nodes[:].data['preprocess']), F.asnumpy(tmp))
        g._sync_barrier(60)
        check_array_shared_memory(g, worker_id, [g.nodes[:].data['preprocess']])
        g._sync_barrier(60)

        # Test apply nodes.
        data = g.nodes[:].data['feat']
        g.apply_nodes(func=lambda nodes: {'feat': F.ones((1, in_feats)) * 10}, v=0)
        assert_almost_equal(F.asnumpy(data[0]), np.squeeze(F.asnumpy(g.nodes[0].data['feat'])))

        # Test apply edges.
        data = g.edges[:].data['feat']
        g.apply_edges(func=lambda edges: {'feat': F.ones((1, in_feats)) * 10}, edges=0)
        assert_almost_equal(F.asnumpy(data[0]), np.squeeze(F.asnumpy(g.edges[0].data['feat'])))

        g.init_ndata('tmp', (g.number_of_nodes(), 10), 'float32')
        data = g.nodes[:].data['tmp']
        # Test pull
        g.pull(1, fn.copy_src(src='feat', out='m'), fn.sum(msg='m', out='tmp'))
        assert_almost_equal(F.asnumpy(data[1]), np.squeeze(F.asnumpy(g.nodes[1].data['preprocess'])))

        # Test send_and_recv
        in_edges = g.in_edges(v=2)
        g.send_and_recv(in_edges, fn.copy_src(src='feat', out='m'), fn.sum(msg='m', out='tmp'))
        assert_almost_equal(F.asnumpy(data[2]), np.squeeze(F.asnumpy(g.nodes[2].data['preprocess'])))

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

    # make server init before worker
    server_init = Value('i', 0)
    serv_p = Process(target=server_func, args=(2, 'test_graph3', server_init))
    serv_p.start()
    while server_init.value == 0:
      time.sleep(1)
    work_p1 = Process(target=check_compute_func, args=(0, 'test_graph3', return_dict))
    work_p2 = Process(target=check_compute_func, args=(1, 'test_graph3', return_dict))
    work_p1.start()
    work_p2.start()
    serv_p.join()
    work_p1.join()
    work_p2.join()
    for worker_id in return_dict.keys():
        assert return_dict[worker_id] == 0, "worker %d fails" % worker_id

def check_sync_barrier(worker_id, graph_name, return_dict):
    try:
        g = create_graph_store(graph_name)
        if g is None:
            return_dict[worker_id] = -1
            return

        if worker_id == 1:
            g.destroy()
            return_dict[worker_id] = 0
            return

        start = time.time()
        try:
            g._sync_barrier(10)
        except TimeoutError as e:
            # this is very loose.
            print("timeout: " + str(abs(time.time() - start)), file=sys.stderr)
            assert 5 < abs(time.time() - start) < 15
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

    # make server init before worker
    server_init = Value('i', 0)
    serv_p = Process(target=server_func, args=(2, 'test_graph4', server_init))
    serv_p.start()
    while server_init.value == 0:
      time.sleep(1)
    work_p1 = Process(target=check_sync_barrier, args=(0, 'test_graph4', return_dict))
    work_p2 = Process(target=check_sync_barrier, args=(1, 'test_graph4', return_dict))
    work_p1.start()
    work_p2.start()
    serv_p.join()
    work_p1.join()
    work_p2.join()
    for worker_id in return_dict.keys():
        assert return_dict[worker_id] == 0, "worker %d fails" % worker_id

def create_mem(gidx, cond_v, shared_v):
    # serialize create_mem before check_mem
    cond_v.acquire()
    gidx1 = gidx.copyto_shared_mem("in", "test_graph5")
    gidx2 = gidx.copyto_shared_mem("out", "test_graph6")
    shared_v.value = 1;
    cond_v.notify()
    cond_v.release()

    # sync for exit
    cond_v.acquire()
    while shared_v.value == 1:
      cond_v.wait()
    cond_v.release()

def check_mem(gidx, cond_v, shared_v):
    # check_mem should run after create_mem
    cond_v.acquire()
    while shared_v.value == 0:
      cond_v.wait()
    cond_v.release()

    gidx1 = dgl.graph_index.from_shared_mem_csr_matrix("test_graph5", gidx.number_of_nodes(),
                                                       gidx.number_of_edges(), "in", False)
    gidx2 = dgl.graph_index.from_shared_mem_csr_matrix("test_graph6", gidx.number_of_nodes(),
                                                       gidx.number_of_edges(), "out", False)
    in_csr = gidx.adjacency_matrix_scipy(False, "csr")
    out_csr = gidx.adjacency_matrix_scipy(True, "csr")

    in_csr1 = gidx1.adjacency_matrix_scipy(False, "csr")
    assert_array_equal(in_csr.indptr, in_csr1.indptr)
    assert_array_equal(in_csr.indices, in_csr1.indices)
    out_csr1 = gidx1.adjacency_matrix_scipy(True, "csr")
    assert_array_equal(out_csr.indptr, out_csr1.indptr)
    assert_array_equal(out_csr.indices, out_csr1.indices)

    in_csr2 = gidx2.adjacency_matrix_scipy(False, "csr")
    assert_array_equal(in_csr.indptr, in_csr2.indptr)
    assert_array_equal(in_csr.indices, in_csr2.indices)
    out_csr2 = gidx2.adjacency_matrix_scipy(True, "csr")
    assert_array_equal(out_csr.indptr, out_csr2.indptr)
    assert_array_equal(out_csr.indices, out_csr2.indices)

    gidx1 = gidx1.copyto_shared_mem("in", "test_graph5")
    gidx2 = gidx2.copyto_shared_mem("out", "test_graph6")

    #sync for exit
    cond_v.acquire()
    shared_v.value = 0;
    cond_v.notify()
    cond_v.release()

def test_copy_shared_mem():
    csr = (spsp.random(num_nodes, num_nodes, density=0.1, format='csr') != 0).astype(np.int64)
    gidx = dgl.graph_index.create_graph_index(csr, False, True)

    cond_v = Condition()
    shared_v = Value('i', 0)
    p1 = Process(target=create_mem, args=(gidx, cond_v, shared_v))
    p2 = Process(target=check_mem, args=(gidx, cond_v, shared_v))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

if __name__ == '__main__':
    test_copy_shared_mem()
    test_init()
    test_sync_barrier()
    test_compute()
