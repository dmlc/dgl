import os
os.environ['OMP_NUM_THREADS'] = '1'
import dgl
import sys
import numpy as np
import time
import socket
from scipy import sparse as spsp
from numpy.testing import assert_array_equal
from multiprocessing import Process, Manager, Condition, Value
import multiprocessing as mp
from dgl.graph_index import create_graph_index
from dgl.data.utils import load_graphs, save_graphs
from dgl.distributed import DistGraphServer, DistGraph
from dgl.distributed import partition_graph, load_partition, load_partition_book, node_split, edge_split
from dgl.distributed import SparseAdagrad, Embedding
from numpy.testing import assert_almost_equal
import backend as F
import math
import unittest
import pickle

if os.name != 'nt':
    import fcntl
    import struct

def get_local_usable_addr():
    """Get local usable IP and port

    Returns
    -------
    str
        IP address, e.g., '192.168.8.12:50051'
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        sock.connect(('10.255.255.255', 1))
        ip_addr = sock.getsockname()[0]
    except ValueError:
        ip_addr = '127.0.0.1'
    finally:
        sock.close()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    sock.close()

    return ip_addr + ' ' + str(port)

def create_random_graph(n):
    arr = (spsp.random(n, n, density=0.001, format='coo') != 0).astype(np.int64)
    ig = create_graph_index(arr, readonly=True)
    return dgl.DGLGraph(ig)

def run_server(graph_name, server_id, num_clients, shared_mem):
    g = DistGraphServer(server_id, "kv_ip_config.txt", num_clients,
                        '/tmp/dist_graph/{}.json'.format(graph_name),
                        disable_shared_mem=not shared_mem)
    print('start server', server_id)
    g.start()

def emb_init(shape, dtype):
    return F.zeros(shape, dtype, F.cpu())

def rand_init(shape, dtype):
    return F.tensor(np.random.normal(size=shape))

def run_client(graph_name, part_id, num_nodes, num_edges):
    time.sleep(5)
    gpb, graph_name = load_partition_book('/tmp/dist_graph/{}.json'.format(graph_name),
                                          part_id, None)
    g = DistGraph("kv_ip_config.txt", graph_name, gpb=gpb)
    check_dist_graph(g, num_nodes, num_edges)

def check_dist_graph(g, num_nodes, num_edges):
    # Test API
    assert g.number_of_nodes() == num_nodes
    assert g.number_of_edges() == num_edges

    # Test reading node data
    nids = F.arange(0, int(g.number_of_nodes() / 2))
    feats1 = g.ndata['features'][nids]
    feats = F.squeeze(feats1, 1)
    assert np.all(F.asnumpy(feats == nids))

    # Test reading edge data
    eids = F.arange(0, int(g.number_of_edges() / 2))
    feats1 = g.edata['features'][eids]
    feats = F.squeeze(feats1, 1)
    assert np.all(F.asnumpy(feats == eids))

    # Test init node data
    new_shape = (g.number_of_nodes(), 2)
    g.ndata['test1'] = dgl.distributed.DistTensor(g, new_shape, F.int32)
    feats = g.ndata['test1'][nids]
    assert np.all(F.asnumpy(feats) == 0)

    # reference to a one that exists
    test2 = dgl.distributed.DistTensor(g, new_shape, F.float32, 'test2', init_func=rand_init)
    test3 = dgl.distributed.DistTensor(g, new_shape, F.float32, 'test2')
    assert np.all(F.asnumpy(test2[nids]) == F.asnumpy(test3[nids]))

    # create a tensor and destroy a tensor and create it again.
    test3 = dgl.distributed.DistTensor(g, new_shape, F.float32, 'test3', init_func=rand_init)
    del test3
    test3 = dgl.distributed.DistTensor(g, (g.number_of_nodes(), 3), F.float32, 'test3')
    del test3

    # test a persistent tesnor
    test4 = dgl.distributed.DistTensor(g, new_shape, F.float32, 'test4', init_func=rand_init,
                                       persistent=True)
    del test4
    try:
        test4 = dgl.distributed.DistTensor(g, (g.number_of_nodes(), 3), F.float32, 'test4')
        raise Exception('')
    except:
        pass

    # Test sparse emb
    try:
        new_shape = (g.number_of_nodes(), 1)
        emb = Embedding(g, new_shape, 'emb1', emb_init)
        lr = 0.001
        optimizer = SparseAdagrad([emb], lr=lr)
        with F.record_grad():
            feats = emb(nids)
            assert np.all(F.asnumpy(feats) == np.zeros((len(nids), 1)))
            loss = F.sum(feats + 1, 0)
        loss.backward()
        optimizer.step()
        feats = emb(nids)
        assert_almost_equal(F.asnumpy(feats), np.ones((len(nids), 1)) * -lr)
        rest = np.setdiff1d(np.arange(g.number_of_nodes()), F.asnumpy(nids))
        feats1 = emb(rest)
        assert np.all(F.asnumpy(feats1) == np.zeros((len(rest), 1)))

        policy = dgl.distributed.PartitionPolicy('node', g.get_partition_book())
        grad_sum = dgl.distributed.DistTensor(g, (g.number_of_nodes(),), F.float32,
                                              'emb1_sum', policy)
        assert np.all(F.asnumpy(grad_sum[nids]) == np.ones((len(nids), 1)))
        assert np.all(F.asnumpy(grad_sum[rest]) == np.zeros((len(rest), 1)))

        emb = Embedding(g, new_shape, 'emb2', emb_init)
        optimizer = SparseAdagrad([emb], lr=lr)
        with F.record_grad():
            feats1 = emb(nids)
            feats2 = emb(nids)
            feats = F.cat([feats1, feats2], 0)
            assert np.all(F.asnumpy(feats) == np.zeros((len(nids) * 2, 1)))
            loss = F.sum(feats + 1, 0)
        loss.backward()
        optimizer.step()
        feats = emb(nids)
        assert_almost_equal(F.asnumpy(feats), np.ones((len(nids), 1)) * math.sqrt(2) * -lr)
        rest = np.setdiff1d(np.arange(g.number_of_nodes()), F.asnumpy(nids))
        feats1 = emb(rest)
        assert np.all(F.asnumpy(feats1) == np.zeros((len(rest), 1)))
    except NotImplementedError as e:
        pass

    # Test write data
    new_feats = F.ones((len(nids), 2), F.int32, F.cpu())
    g.ndata['test1'][nids] = new_feats
    feats = g.ndata['test1'][nids]
    assert np.all(F.asnumpy(feats) == 1)

    # Test metadata operations.
    assert len(g.ndata['features']) == g.number_of_nodes()
    assert g.ndata['features'].shape == (g.number_of_nodes(), 1)
    assert g.ndata['features'].dtype == F.int64
    assert g.node_attr_schemes()['features'].dtype == F.int64
    assert g.node_attr_schemes()['test1'].dtype == F.int32
    assert g.node_attr_schemes()['features'].shape == (1,)

    selected_nodes = np.random.randint(0, 100, size=g.number_of_nodes()) > 30
    # Test node split
    nodes = node_split(selected_nodes, g.get_partition_book())
    nodes = F.asnumpy(nodes)
    # We only have one partition, so the local nodes are basically all nodes in the graph.
    local_nids = np.arange(g.number_of_nodes())
    for n in nodes:
        assert n in local_nids

    print('end')

def check_server_client(shared_mem):
    prepare_dist()
    g = create_random_graph(10000)

    # Partition the graph
    num_parts = 1
    graph_name = 'dist_graph_test_2'
    g.ndata['features'] = F.unsqueeze(F.arange(0, g.number_of_nodes()), 1)
    g.edata['features'] = F.unsqueeze(F.arange(0, g.number_of_edges()), 1)
    partition_graph(g, graph_name, num_parts, '/tmp/dist_graph')

    # let's just test on one partition for now.
    # We cannot run multiple servers and clients on the same machine.
    serv_ps = []
    ctx = mp.get_context('spawn')
    for serv_id in range(1):
        p = ctx.Process(target=run_server, args=(graph_name, serv_id, 1, shared_mem))
        serv_ps.append(p)
        p.start()

    cli_ps = []
    for cli_id in range(1):
        print('start client', cli_id)
        p = ctx.Process(target=run_client, args=(graph_name, cli_id, g.number_of_nodes(),
                                                 g.number_of_edges()))
        p.start()
        cli_ps.append(p)

    for p in cli_ps:
        p.join()

    for p in serv_ps:
        p.join()

    print('clients have terminated')

@unittest.skipIf(dgl.backend.backend_name == "tensorflow", reason="TF doesn't support some of operations in DistGraph")
def test_server_client():
    os.environ['DGL_DIST_MODE'] = 'distributed'
    check_server_client(True)
    check_server_client(False)

@unittest.skipIf(dgl.backend.backend_name == "tensorflow", reason="TF doesn't support some of operations in DistGraph")
def test_standalone():
    os.environ['DGL_DIST_MODE'] = 'standalone'
    g = create_random_graph(10000)
    # Partition the graph
    num_parts = 1
    graph_name = 'dist_graph_test_3'
    g.ndata['features'] = F.unsqueeze(F.arange(0, g.number_of_nodes()), 1)
    g.edata['features'] = F.unsqueeze(F.arange(0, g.number_of_edges()), 1)
    partition_graph(g, graph_name, num_parts, '/tmp/dist_graph')
    dist_g = DistGraph("kv_ip_config.txt", graph_name,
                  conf_file='/tmp/dist_graph/{}.json'.format(graph_name))
    check_dist_graph(dist_g, g.number_of_nodes(), g.number_of_edges())

def test_split():
    prepare_dist()
    g = create_random_graph(10000)
    num_parts = 4
    num_hops = 2
    partition_graph(g, 'dist_graph_test', num_parts, '/tmp/dist_graph', num_hops=num_hops, part_method='metis')

    node_mask = np.random.randint(0, 100, size=g.number_of_nodes()) > 30
    edge_mask = np.random.randint(0, 100, size=g.number_of_edges()) > 30
    selected_nodes = np.nonzero(node_mask)[0]
    selected_edges = np.nonzero(edge_mask)[0]
    for i in range(num_parts):
        dgl.distributed.set_num_client(num_parts)
        part_g, node_feats, edge_feats, gpb, _ = load_partition('/tmp/dist_graph/dist_graph_test.json', i)
        local_nids = F.nonzero_1d(part_g.ndata['inner_node'])
        local_nids = F.gather_row(part_g.ndata[dgl.NID], local_nids)
        nodes1 = np.intersect1d(selected_nodes, F.asnumpy(local_nids))
        nodes2 = node_split(node_mask, gpb, i)
        assert np.all(np.sort(nodes1) == np.sort(F.asnumpy(nodes2)))
        local_nids = F.asnumpy(local_nids)
        for n in nodes1:
            assert n in local_nids

        dgl.distributed.set_num_client(num_parts * 2)
        nodes3 = node_split(node_mask, gpb, i * 2)
        nodes4 = node_split(node_mask, gpb, i * 2 + 1)
        nodes5 = F.cat([nodes3, nodes4], 0)
        assert np.all(np.sort(nodes1) == np.sort(F.asnumpy(nodes5)))

        dgl.distributed.set_num_client(num_parts)
        local_eids = F.nonzero_1d(part_g.edata['inner_edge'])
        local_eids = F.gather_row(part_g.edata[dgl.EID], local_eids)
        edges1 = np.intersect1d(selected_edges, F.asnumpy(local_eids))
        edges2 = edge_split(edge_mask, gpb, i)
        assert np.all(np.sort(edges1) == np.sort(F.asnumpy(edges2)))
        local_eids = F.asnumpy(local_eids)
        for e in edges1:
            assert e in local_eids

        dgl.distributed.set_num_client(num_parts * 2)
        edges3 = edge_split(edge_mask, gpb, i * 2)
        edges4 = edge_split(edge_mask, gpb, i * 2 + 1)
        edges5 = F.cat([edges3, edges4], 0)
        assert np.all(np.sort(edges1) == np.sort(F.asnumpy(edges5)))

def test_split_even():
    prepare_dist()
    g = create_random_graph(10000)
    num_parts = 4
    num_hops = 2
    partition_graph(g, 'dist_graph_test', num_parts, '/tmp/dist_graph', num_hops=num_hops, part_method='metis')

    node_mask = np.random.randint(0, 100, size=g.number_of_nodes()) > 30
    edge_mask = np.random.randint(0, 100, size=g.number_of_edges()) > 30
    selected_nodes = np.nonzero(node_mask)[0]
    selected_edges = np.nonzero(edge_mask)[0]
    all_nodes1 = []
    all_nodes2 = []
    all_edges1 = []
    all_edges2 = []
    for i in range(num_parts):
        dgl.distributed.set_num_client(num_parts)
        part_g, node_feats, edge_feats, gpb, _ = load_partition('/tmp/dist_graph/dist_graph_test.json', i)
        local_nids = F.nonzero_1d(part_g.ndata['inner_node'])
        local_nids = F.gather_row(part_g.ndata[dgl.NID], local_nids)
        nodes = node_split(node_mask, gpb, i, force_even=True)
        all_nodes1.append(nodes)
        subset = np.intersect1d(F.asnumpy(nodes), F.asnumpy(local_nids))
        print('part {} get {} nodes and {} are in the partition'.format(i, len(nodes), len(subset)))

        dgl.distributed.set_num_client(num_parts * 2)
        nodes1 = node_split(node_mask, gpb, i * 2, force_even=True)
        nodes2 = node_split(node_mask, gpb, i * 2 + 1, force_even=True)
        nodes3 = F.cat([nodes1, nodes2], 0)
        all_nodes2.append(nodes3)
        subset = np.intersect1d(F.asnumpy(nodes), F.asnumpy(nodes3))
        print('intersection has', len(subset))

        dgl.distributed.set_num_client(num_parts)
        local_eids = F.nonzero_1d(part_g.edata['inner_edge'])
        local_eids = F.gather_row(part_g.edata[dgl.EID], local_eids)
        edges = edge_split(edge_mask, gpb, i, force_even=True)
        all_edges1.append(edges)
        subset = np.intersect1d(F.asnumpy(edges), F.asnumpy(local_eids))
        print('part {} get {} edges and {} are in the partition'.format(i, len(edges), len(subset)))

        dgl.distributed.set_num_client(num_parts * 2)
        edges1 = edge_split(edge_mask, gpb, i * 2, force_even=True)
        edges2 = edge_split(edge_mask, gpb, i * 2 + 1, force_even=True)
        edges3 = F.cat([edges1, edges2], 0)
        all_edges2.append(edges3)
        subset = np.intersect1d(F.asnumpy(edges), F.asnumpy(edges3))
        print('intersection has', len(subset))
    all_nodes1 = F.cat(all_nodes1, 0)
    all_edges1 = F.cat(all_edges1, 0)
    all_nodes2 = F.cat(all_nodes2, 0)
    all_edges2 = F.cat(all_edges2, 0)
    all_nodes = np.nonzero(node_mask)[0]
    all_edges = np.nonzero(edge_mask)[0]
    assert np.all(all_nodes == F.asnumpy(all_nodes1))
    assert np.all(all_edges == F.asnumpy(all_edges1))
    assert np.all(all_nodes == F.asnumpy(all_nodes2))
    assert np.all(all_edges == F.asnumpy(all_edges2))

def prepare_dist():
    ip_config = open("kv_ip_config.txt", "w")
    ip_addr = get_local_usable_addr()
    ip_config.write('%s 1\n' % ip_addr)
    ip_config.close()

if __name__ == '__main__':
    os.makedirs('/tmp/dist_graph', exist_ok=True)
    test_split()
    test_split_even()
    test_server_client()
    test_standalone()
