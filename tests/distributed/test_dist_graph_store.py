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
from dgl.contrib import DistGraphStoreServer, DistGraphStore
import backend as F
import unittest

server_namebook = {0: [0, '127.0.0.1', 30000, 1]}

def create_random_graph(n):
    arr = (spsp.random(n, n, density=0.001, format='coo') != 0).astype(np.int64)
    ig = create_graph_index(arr, multigraph=False, readonly=True)
    return dgl.DGLGraph(ig)

def run_server(graph_name, server_id, num_clients, barrier):
    server_data = './server-' + str(server_id) + '.dgl'
    client_data = './client-' + str(server_id) + '.dgl'
    g = DistGraphStoreServer(server_namebook, server_id, graph_name, server_data, client_data, num_clients)
    barrier.wait()
    print('start server', server_id)
    g.start()

def run_client(graph_name, part, barrier):
    barrier.wait()
    g = DistGraphStore(server_namebook, graph_name)
    print('|V|={}, |E|={}, #local:{}'.format(g.g.number_of_nodes(), g.g.number_of_edges(), len(g.local_nids)))
    assert part.number_of_nodes() == g.g.number_of_nodes()
    assert part.number_of_edges() == g.g.number_of_edges()
    assert np.all(F.asnumpy(part.ndata['part_id'] == g.g.ndata['part_id']))
    print('end')

def test_create():
    g = create_random_graph(10000)

    # Partition the graph
    num_parts = 4
    node_parts = dgl.transform.metis_partition_assignment(g, num_parts)
    server_parts = dgl.transform.partition_graph_with_halo(g, node_parts, 0)
    client_parts = dgl.transform.partition_graph_with_halo(g, node_parts, 1)
    for part_id in client_parts:
        part = client_parts[part_id]
        part.ndata['part_id'] = F.gather_row(node_parts, part.ndata[dgl.NID])

    # save the partitions to files.
    for part_id in range(num_parts):
        serv_part = server_parts[part_id]
        part = client_parts[part_id]
        save_graphs('./server-' + str(part_id) + '.dgl', [serv_part])
        save_graphs('./client-' + str(part_id) + '.dgl', [part])

    graph_name = 'test'
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
        p = Process(target=run_client, args=(graph_name, client_parts[cli_id], barrier))
        p.start()
        cli_ps.append(p)

    #for p in serv_ps:
    #    p.join()
    for p in cli_ps:
        p.join()
    print('clients have terminated')

if __name__ == '__main__':
    test_create()
