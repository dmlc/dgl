import os
os.environ['OMP_NUM_THREADS'] = '1'
import dgl
import sys
import numpy as np
import time
from scipy import sparse as spsp
from numpy.testing import assert_array_equal
from multiprocessing import Process, Manager, Condition, Value
import multiprocessing as mp
from dgl.graph_index import create_graph_index
from dgl.data.utils import load_graphs, save_graphs
from dgl.distributed import DistGraphServer, DistGraph
from dgl.distributed import partition_graph, load_partition, GraphPartitionBook, node_split, edge_split
import backend as F
import unittest
import pickle

server_namebook = {0: [0, '127.0.0.1', 30000, 1]}

def create_random_graph(n):
    arr = (spsp.random(n, n, density=0.001, format='coo') != 0).astype(np.int64)
    ig = create_graph_index(arr, readonly=True)
    return dgl.DGLGraph(ig)

def run_server(graph_name, server_id, num_clients, barrier):
    g = DistGraphServer(server_id, server_namebook, num_clients, graph_name,
                        '/tmp/{}.json'.format(graph_name))
    barrier.wait()
    print('start server', server_id)
    g.start()

def run_client(graph_name, barrier, num_nodes, num_edges):
    barrier.wait()
    g = DistGraph(server_namebook, graph_name)

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
    g.init_ndata('test1', new_shape, F.int32)
    feats = g.ndata['test1'][nids]
    assert np.all(F.asnumpy(feats) == 0)

    # Test init edge data
    new_shape = (g.number_of_edges(), 2)
    g.init_edata('test1', new_shape, F.int32)
    feats = g.edata['test1'][eids]
    assert np.all(F.asnumpy(feats) == 0)

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
    nodes = node_split(selected_nodes, g.get_partition_book(), g.rank())
    nodes = F.asnumpy(nodes)
    # We only have one partition, so the local nodes are basically all nodes in the graph.
    local_nids = np.arange(g.number_of_nodes())
    for n in nodes:
        assert n in local_nids

    g.shut_down()
    print('end')

@unittest.skipIf(dgl.backend.backend_name == "tensorflow", reason="TF doesn't support some of operations in DistGraph")
def test_server_client():
    g = create_random_graph(10000)

    # Partition the graph
    num_parts = 1
    graph_name = 'test'
    g.ndata['features'] = F.unsqueeze(F.arange(0, g.number_of_nodes()), 1)
    g.edata['features'] = F.unsqueeze(F.arange(0, g.number_of_edges()), 1)
    partition_graph(g, graph_name, num_parts, '/tmp')

    # let's just test on one partition for now.
    # We cannot run multiple servers and clients on the same machine.
    barrier = mp.Barrier(2)
    serv_ps = []
    for serv_id in range(1):
        p = Process(target=run_server, args=(graph_name, serv_id, 1, barrier))
        serv_ps.append(p)
        p.start()

    cli_ps = []
    for cli_id in range(1):
        print('start client', cli_id)
        p = Process(target=run_client, args=(graph_name, barrier, g.number_of_nodes(),
                                             g.number_of_edges()))
        p.start()
        cli_ps.append(p)

    for p in cli_ps:
        p.join()
    print('clients have terminated')

def test_split():
    g = create_random_graph(10000)
    num_parts = 4
    num_hops = 2
    partition_graph(g, 'test', num_parts, '/tmp', num_hops=num_hops, part_method='metis')

    node_mask = np.random.randint(0, 100, size=g.number_of_nodes()) > 30
    edge_mask = np.random.randint(0, 100, size=g.number_of_edges()) > 30
    selected_nodes = np.nonzero(node_mask)[0]
    selected_edges = np.nonzero(edge_mask)[0]
    for i in range(num_parts):
        part_g, node_feats, edge_feats, meta = load_partition('/tmp/test.json', i)
        num_nodes, num_edges, node_map, edge_map, num_partitions = meta
        gpb = GraphPartitionBook(part_id=i,
                                 num_parts=num_partitions,
                                 node_map=node_map,
                                 edge_map=edge_map,
                                 part_graph=part_g)
        local_nids = F.nonzero_1d(part_g.ndata['local_node'])
        local_nids = F.gather_row(part_g.ndata[dgl.NID], local_nids)
        nodes1 = np.intersect1d(selected_nodes, F.asnumpy(local_nids))
        nodes2 = node_split(node_mask, gpb, i)
        assert np.all(np.sort(nodes1) == np.sort(F.asnumpy(nodes2)))
        local_nids = F.asnumpy(local_nids)
        for n in nodes1:
            assert n in local_nids

        local_eids = F.nonzero_1d(part_g.edata['local_edge'])
        local_eids = F.gather_row(part_g.edata[dgl.EID], local_eids)
        edges1 = np.intersect1d(selected_edges, F.asnumpy(local_eids))
        edges2 = edge_split(edge_mask, gpb, i)
        assert np.all(np.sort(edges1) == np.sort(F.asnumpy(edges2)))
        local_eids = F.asnumpy(local_eids)
        for e in edges1:
            assert e in local_eids

if __name__ == '__main__':
    test_split()
    test_server_client()
