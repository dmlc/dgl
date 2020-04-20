import backend as F
import numpy as np
import scipy as sp
import dgl
from dgl import utils
from dgl.contrib import KVServer
from dgl.contrib import KVClient
from numpy.testing import assert_array_equal

import os
import time

num_entries = 10
dim_size = 3

server_namebook = {0:[0, '127.0.0.1', 30070, 1]}

data_0 = F.zeros((num_entries, dim_size), F.float32, F.cpu())
g2l_0 = F.arange(0, num_entries)
partition_0 = F.zeros(num_entries, F.int64, F.cpu())

data_1 = F.zeros((num_entries*2, dim_size), F.float32, F.cpu())
g2l_1 = F.arange(0, num_entries*2)
partition_1 = F.zeros(num_entries*2, F.int64, F.cpu())

def start_server():
    my_server = KVServer(server_id=0, server_namebook=server_namebook, num_client=1)

    my_server.set_global2local(name='data_0', global2local=g2l_0)
    my_server.set_global2local(name='data_1', global2local=g2l_1)
    my_server.set_partition_book(name='data_0', partition_book=partition_0)
    my_server.set_partition_book(name='data_1', partition_book=partition_1)
    my_server.init_data(name='data_0', data_tensor=data_0)
    my_server.init_data(name='data_1', data_tensor=data_1)

    my_server.start()


def start_client():
    my_client = KVClient(server_namebook=server_namebook)
    my_client.connect()

    name_list = my_client.get_data_name_list()
    assert len(name_list) == 2
    assert 'data_0' in name_list
    assert 'data_1' in name_list

    meta_0 = my_client.get_data_meta('data_0')
    assert meta_0[0] == F.float32
    assert_array_equal(meta_0[2], partition_0)

    meta_1 = my_client.get_data_meta('data_1')
    assert meta_1[0] == F.float32
    assert_array_equal(meta_1[2], partition_1)

    my_client.push(name='data_0', id_tensor=F.tensor([0, 1, 2]), data_tensor=F.tensor([[1.,1.,1.],[2.,2.,2.],[3.,3.,3.]]))

    res = my_client.pull(name='data_0', id_tensor=F.tensor([0, 1, 2]))

    target = F.tensor([[1.,1.,1.],[2.,2.,2.],[3.,3.,3.]])

    assert_array_equal(res, target)

    my_client.shut_down()


if __name__ == '__main__':
    pid = os.fork()
    if pid == 0:
        start_server()
    else:
        time.sleep(2) # wait trainer start
        start_client()
