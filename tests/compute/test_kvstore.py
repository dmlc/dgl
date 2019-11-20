import backend as F
import numpy as np
import scipy as sp
import dgl
import torch as th
from dgl import utils

import os
import time

client_namebook = { 0:'127.0.0.1:50061' }

server_namebook = { 0:'127.0.0.1:50062' }

def start_server(server_embed):
    server = dgl.contrib.KVServer(
        server_id=0, 
        client_namebook=client_namebook, 
        server_addr=server_namebook[0])

    server.init_data(name='server_embed', data_tensor=server_embed)

    server.start()

def start_client(server_embed):
    client = dgl.contrib.KVClient(
        client_id=0, 
        server_namebook=server_namebook, 
        client_addr=client_namebook[0])

    client.connect()

    # Initialize data on server
    client.init_data(name='embed_0', server_id=0, shape=[5, 3], init_type='zero')
    client.init_data(name='embed_1', server_id=0, shape=[5], init_type='uniform', low=0.0, high=0.0)

    data_0 = th.tensor([[0., 0., 0., ], [1., 1., 1.], [2., 2., 2.]])
    data_1 = th.tensor([0., 1., 2.])

    for i in range(5):
        client.push(name='embed_0', server_id=0, id_tensor=th.tensor([0, 2, 4]), data_tensor=data_0)
        client.push(name='embed_1', server_id=0, id_tensor=th.tensor([0, 2, 4]), data_tensor=data_1)
        client.push(name='server_embed', server_id=0, id_tensor=th.tensor([0, 2, 4]), data_tensor=data_1)

    client.barrier()

    client.pull(name='embed_0', server_id=0, id_tensor=th.tensor([0, 1, 2, 3, 4]))
    msg = client.pull_wait()
    assert msg.rank == 0

    target_tensor_0 = th.tensor(
        [[ 0., 0., 0.],
         [ 0., 0., 0.],
         [ 5., 5., 5.],
         [ 0., 0., 0.],
         [10., 10., 10.]])

    assert th.equal(msg.data, target_tensor_0) == True

    client.pull(name='embed_1', server_id=0, id_tensor=th.tensor([0, 1, 2, 3, 4]))
    msg = client.pull_wait()

    target_tensor_1 = th.tensor([ 0., 0., 5., 0., 10.])

    assert th.equal(msg.data, target_tensor_1) == True

    client.pull(name='embed_0', server_id=0, id_tensor=th.tensor([0, 1, 2, 3, 4]))
    client.pull(name='embed_1', server_id=0, id_tensor=th.tensor([0, 1, 2, 3, 4]))
    client.pull(name='embed_0', server_id=0, id_tensor=th.tensor([0, 1, 2, 3, 4]))
    client.pull(name='embed_1', server_id=0, id_tensor=th.tensor([0, 1, 2, 3, 4]))
    client.pull(name='server_embed', server_id=0, id_tensor=th.tensor([0, 1, 2, 3, 4]))

    msg_0 = client.pull_wait()
    msg_1 = client.pull_wait()
    msg_2 = client.pull_wait()
    msg_3 = client.pull_wait()
    msg_4 = client.pull_wait()

    target_tensor_2 = th.tensor([ 2., 2., 7., 2., 12.])

    assert th.equal(msg_0.data, target_tensor_0) == True
    assert th.equal(msg_1.data, target_tensor_1) == True
    assert th.equal(msg_2.data, target_tensor_0) == True
    assert th.equal(msg_3.data, target_tensor_1) == True
    assert th.equal(msg_4.data, target_tensor_2) == True

    server_embed += target_tensor_2

    client.pull(name='server_embed', server_id=0, id_tensor=th.tensor([0, 1, 2, 3, 4]))
    msg_5 = client.pull_wait()

    assert th.equal(msg_5.data, target_tensor_2 * 2) == True

    client.shut_down()

if __name__ == '__main__':
    server_embed = th.tensor([2., 2., 2., 2., 2.])
    # use pytorch shared memory
    server_embed.share_memory_()

    pid = os.fork()
    if pid == 0:
        start_server(server_embed)
    else:
        time.sleep(2) # wait server start
        start_client(server_embed)

    assert th.equal(server_embed, th.tensor([ 4., 4., 14., 4., 24.])) == True