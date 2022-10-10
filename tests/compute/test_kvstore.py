import os
import time

import backend as F
import numpy as np
import scipy as sp
from numpy.testing import assert_array_equal

import dgl
from dgl import utils
from dgl.contrib import KVClient, KVServer

num_entries = 10
dim_size = 3

server_namebook = {0: [0, "127.0.0.1", 30070, 1]}

data_0 = F.zeros((num_entries, dim_size), F.float32, F.cpu())
g2l_0 = F.arange(0, num_entries)
partition_0 = F.zeros(num_entries, F.int64, F.cpu())

data_1 = F.zeros((num_entries * 2, dim_size), F.float32, F.cpu())
g2l_1 = F.arange(0, num_entries * 2)
partition_1 = F.zeros(num_entries * 2, F.int64, F.cpu())

data_3 = F.zeros((num_entries, dim_size), F.int64, F.cpu())
data_4 = F.zeros((num_entries, dim_size), F.float64, F.cpu())
data_5 = F.zeros((num_entries, dim_size), F.int32, F.cpu())


def start_server():
    my_server = KVServer(
        server_id=0, server_namebook=server_namebook, num_client=1
    )
    my_server.set_global2local(name="data_0", global2local=g2l_0)
    my_server.set_global2local(name="data_1", global2local=g2l_1)
    my_server.set_global2local(name="data_3", global2local=g2l_0)
    my_server.set_global2local(name="data_4", global2local=g2l_0)
    my_server.set_global2local(name="data_5", global2local=g2l_0)
    my_server.set_partition_book(name="data_0", partition_book=partition_0)
    my_server.set_partition_book(name="data_1", partition_book=partition_1)
    my_server.set_partition_book(name="data_3", partition_book=partition_0)
    my_server.set_partition_book(name="data_4", partition_book=partition_0)
    my_server.set_partition_book(name="data_5", partition_book=partition_0)
    my_server.init_data(name="data_0", data_tensor=data_0)
    my_server.init_data(name="data_1", data_tensor=data_1)
    my_server.init_data(name="data_3", data_tensor=data_3)
    my_server.init_data(name="data_4", data_tensor=data_4)
    my_server.init_data(name="data_5", data_tensor=data_5)

    my_server.start()


def start_client():
    my_client = KVClient(server_namebook=server_namebook)
    my_client.connect()

    my_client.init_data(
        name="data_2",
        shape=(num_entries, dim_size),
        dtype=F.float32,
        target_name="data_0",
    )
    print("Init data from client..")

    name_list = my_client.get_data_name_list()
    assert len(name_list) == 6
    assert "data_0" in name_list
    assert "data_1" in name_list
    assert "data_2" in name_list
    assert "data_3" in name_list
    assert "data_4" in name_list
    assert "data_5" in name_list

    meta_0 = my_client.get_data_meta("data_0")
    assert meta_0[0] == F.float32
    assert meta_0[1] == tuple(F.shape(data_0))
    assert_array_equal(meta_0[2], partition_0)

    meta_1 = my_client.get_data_meta("data_1")
    assert meta_1[0] == F.float32
    assert meta_1[1] == tuple(F.shape(data_1))
    assert_array_equal(meta_1[2], partition_1)

    meta_2 = my_client.get_data_meta("data_2")
    assert meta_2[0] == F.float32
    assert meta_2[1] == tuple(F.shape(data_0))
    assert_array_equal(meta_2[2], partition_0)

    meta_3 = my_client.get_data_meta("data_3")
    assert meta_3[0] == F.int64
    assert meta_3[1] == tuple(F.shape(data_3))
    assert_array_equal(meta_3[2], partition_0)

    meta_4 = my_client.get_data_meta("data_4")
    assert meta_4[0] == F.float64
    assert meta_4[1] == tuple(F.shape(data_4))
    assert_array_equal(meta_3[2], partition_0)

    meta_5 = my_client.get_data_meta("data_5")
    assert meta_5[0] == F.int32
    assert meta_5[1] == tuple(F.shape(data_5))
    assert_array_equal(meta_3[2], partition_0)

    my_client.push(
        name="data_0",
        id_tensor=F.tensor([0, 1, 2]),
        data_tensor=F.tensor(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
        ),
    )
    my_client.push(
        name="data_2",
        id_tensor=F.tensor([0, 1, 2]),
        data_tensor=F.tensor(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
        ),
    )
    my_client.push(
        name="data_3",
        id_tensor=F.tensor([0, 1, 2]),
        data_tensor=F.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
    )
    my_client.push(
        name="data_4",
        id_tensor=F.tensor([0, 1, 2]),
        data_tensor=F.tensor(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], F.float64
        ),
    )
    my_client.push(
        name="data_5",
        id_tensor=F.tensor([0, 1, 2]),
        data_tensor=F.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]], F.int32),
    )

    target = F.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])

    res = my_client.pull(name="data_0", id_tensor=F.tensor([0, 1, 2]))
    assert_array_equal(res, target)

    res = my_client.pull(name="data_2", id_tensor=F.tensor([0, 1, 2]))
    assert_array_equal(res, target)

    target = F.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    res = my_client.pull(name="data_3", id_tensor=F.tensor([0, 1, 2]))
    assert_array_equal(res, target)

    target = F.tensor(
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], F.float64
    )

    res = my_client.pull(name="data_4", id_tensor=F.tensor([0, 1, 2]))
    assert_array_equal(res, target)

    target = F.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]], F.int32)

    res = my_client.pull(name="data_5", id_tensor=F.tensor([0, 1, 2]))
    assert_array_equal(res, target)

    my_client.shut_down()


if __name__ == "__main__":
    pid = os.fork()
    if pid == 0:
        start_server()
    else:
        time.sleep(2)  # wait trainer start
        start_client()
