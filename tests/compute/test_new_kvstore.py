import os
import time

import dgl
import backend as F
import unittest, pytest

from numpy.testing import assert_array_equal

def start_server():
    server_state = dgl.distributed.ServerState(None)
    dgl.distributed.start_server(server_id=0, ip_config='ip_config.txt', num_clients=1)

def start_client():
    dgl.distributed.connect_to_server(ip_config='ip_config.txt')
    # clean up
    dgl.distributed.shutdown_servers()
    dgl.distributed.finalize_client()

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
def test_kv_store():
    ip_config = open("ip_config.txt", "w")
    ip_config.write('127.0.0.1 30050 1\n')
    ip_config.close()
    pid = os.fork()
    if pid == 0:
        start_server()
    else:
        time.sleep(1)
        start_client()

if __name__ == '__main__':
    test_kv_store()