import dgl
import unittest
import os
from dgl.data import CitationGraphDataset
from dgl.distributed.sampling import sample_neighbors
from dgl.distributed import partition_graph, load_partition, GraphPartitionBook
import sys
import multiprocessing as mp
import torch as th
import time


def myexcepthook(exctype, value, traceback):
    for p in mp.active_children():
        p.terminate()


class MochDistGraph:

    def __init__(self, partition_book):
        self.partition_book = partition_book

    def get_partition_book(self):
        return self.partition_book


class MockServerState:

    def __init__(self, g):
        self.state = dgl.distributed.ServerState(None)
        self.hgraph = dgl.as_heterograph(g)

    @property
    def graph(self):
        return self.hgraph

    @property
    def total_num_nodes(self):
        return self.hgraph.number_of_nodes()

    @property
    def total_num_edges(self):
        return self.hgraph.number_of_edges()


def start_server(rank):
    import dgl
    part_g, node_feats, edge_feats, meta = load_partition(
        '/tmp/test.json', rank)
    num_nodes, num_edges, node_map, edge_map, num_partitions = meta
    gpb = GraphPartitionBook(part_id=rank,
                             num_parts=num_partitions,
                             node_map=node_map,
                             edge_map=edge_map,
                             part_graph=part_g)
    server_state = MockServerState(part_g)
    server_state.rank = rank
    server_state.partition_book = gpb
    dgl.distributed.start_server(server_id=rank,
                                 ip_config='rpc_ip_config.txt',
                                 num_clients=1,
                                 server_state=server_state)
    pass


def start_client(rank):
    import dgl
    dgl.distributed.connect_to_server(ip_config='rpc_ip_config.txt')

    part_g, node_feats, edge_feats, meta = load_partition(
        '/tmp/test.json', rank)
    num_nodes, num_edges, node_map, edge_map, num_partitions = meta
    print(node_map)
    gpb = GraphPartitionBook(part_id=rank,
                             num_parts=num_partitions,
                             node_map=node_map,
                             edge_map=edge_map,
                             part_graph=part_g)

    g = MochDistGraph(gpb)
    results = sample_neighbors(g, [0], 1)
    print(results)
    print("11111111")
    dgl.distributed.shutdown_servers()
    dgl.distributed.finalize_client()


@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
def test_rpc():
    num_server = 3
    ip_config = open("rpc_ip_config.txt", "w")
    ip_config.write(f'127.0.0.1 30050 {num_server}\n')
    ip_config.close()

    # partition graph
    g = CitationGraphDataset("cora")[0]
    g.readonly()
    g.ndata['labels'] = th.arange(0, g.number_of_nodes())
    # g.ndata['feats'] = F.tensor(np.random.randn(g.number_of_nodes(), 10))
    num_parts = num_server
    num_hops = 2

    partition_graph(g, 'test', num_parts, '/tmp',
                    num_hops=num_hops, part_method='metis')

    pserver_list = []
    mp.set_start_method("spawn")
    for i in range(num_server):
        p = mp.Process(target=start_server, args=(i,))
        p.start()
        # time.sleep(1)
        pserver_list.append(p)

    pclient = mp.Process(target=start_client, args=(i,))
    sys.excepthook = myexcepthook
    pclient.start()
    pclient.join()
    for p in pserver_list:
        p.join()


if __name__ == "__main__":
    test_rpc()
