import dgl
import unittest
import os
from dgl.data import CitationGraphDataset
import sys
import multiprocessing as mp


def myexcepthook(exctype, value, traceback):
    for p in mp.active_children():
       p.terminate()


class MockServerState:

    def __init__(self):
        self.state = dgl.distributed.ServerState(None)
        g = CitationGraphDataset("cora")[0]
        g.readonly()
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
    server_state = MockServerState()
    dgl.distributed.start_server(server_id=rank,
                                 ip_config='rpc_ip_config.txt',
                                 num_clients=1,
                                 server_state=server_state)
    pass


def start_client(rank):
    import dgl
    dgl.distributed.connect_to_server(ip_config='rpc_ip_config.txt')

    dgl.distributed.shutdown_servers()
    dgl.distributed.finalize_client()


@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
def test_rpc():
    num_server = 1
    ip_config = open("rpc_ip_config.txt", "w")
    ip_config.write(f'127.0.0.1 30060 {num_server}\n')
    ip_config.close()

    pserver_list = []
    mp.set_start_method("spawn")
    for i in range(num_server):
        p = mp.Process(target=start_server, args=(i,))
        p.start()
        pserver_list.append(p)

    pclient = mp.Process(target=start_client, args=(i,))
    sys.excepthook = myexcepthook
    pclient.start()
    pclient.join()
    for p in pserver_list:
        p.join()



if __name__ == "__main__":
    test_rpc()
