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

def start_server():
    server = dgl.contrib.KVServer(
        server_id=0, 
        client_namebook=client_namebook, 
        server_addr=server_namebook[0])

    server.start()

def start_client():
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

    client.barrier()

    client.pull(name='embed_0', server_id=0, id_tensor=th.tensor([0, 1, 2, 3, 4]))
    server_id, new_tensor = client.pull_wait()
    assert server_id == 0

    target_tensor = th.tensor(
        [[ 0.  0.  0.],
         [ 0.  0.  0.],
         [20. 20. 20.],
         [ 0.  0.  0.],
         [40. 40. 40.]])

    assert th.equal(new_tensor, target_tensor) == True

    client.pull(name='embed_1', server_id=0, id_tensor=th.tensor([0, 1, 2, 3, 4]))
    server_id, new_tensor = client.pull_wait()

    target_tensor = th.tensor([ 0., 0., 20., 0., 40.])

    assert th.equal(new_tensor, target_tensor) == True

    client.shut_down()

if __name__ == '__main__':
    pid = os.fork()
    if pid == 0:
        start_server()
    else:
        time.sleep(2) # wait server start
        start_client()
