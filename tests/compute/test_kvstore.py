import backend as F
import numpy as np
import scipy as sp
import dgl
import torch
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

    client.init_data(name='embed_0', shape=[10, 3], init_type='zero')
    client.init_data(name='embed_1', shape=[11, 3], init_type='uniform', low=0.0, high=0.0)
    client.init_data(name='embed_2', shape=[11], init_type='zero')

    tensor_id = torch.tensor([0, 1, 2])
    tensor_data = torch.tensor([[0., 0., 0., ], [1., 1., 1.], [2., 2., 2.]])

    # Push
    for i in range(5):
        client.push('embed_0', tensor_id, tensor_data)
        client.push('embed_1', tensor_id, tensor_data)
        client.push('embed_2', tensor_id, torch.tensor([2., 2., 2.]))

    tensor_id = torch.tensor([6, 7, 8])
    for i in range(5):
        client.push('embed_0', tensor_id, tensor_data)
        client.push('embed_1', tensor_id, tensor_data)
        client.push('embed_2', tensor_id, torch.tensor([3., 3., 3.]))

    # Pull
    tensor_id = torch.tensor([0, 1, 2, 6, 7, 8])
    new_tensor_0 = client.pull('embed_0', tensor_id)
    new_tensor_1 = client.pull('embed_1', tensor_id)
    new_tensor_2 = client.pull('embed_2', tensor_id)
    
    target_tensor = torch.tensor(
        [[ 0.,  0.,  0.],
        [ 5.,  5.,  5.],
        [10., 10., 10.],
        [ 0.,  0.,  0.],
        [ 5.,  5.,  5.],
        [10., 10., 10.]])

    assert torch.equal(new_tensor_0, target_tensor) == True
    assert torch.equal(new_tensor_1, target_tensor) == True

    target_tensor = tensor.tensor([10., 10., 10., 15., 15., 15.])

    assert torch.equal(new_tensor_2, target_tensor) == True

    client.push_all('embed_0', client.pull_all('embed_0'))
    client.push_all('embed_1', client.pull_all('embed_1'))
    client.push_all('embed_2', client.pull_all('embed_2'))

    # Pull
    tensor_id = torch.tensor([0, 1, 2, 6, 7, 8])
    new_tensor_0 = client.pull('embed_0', tensor_id)
    new_tensor_1 = client.pull('embed_1', tensor_id)
    new_tensor_2 = client.pull('embed_2', tensor_id)

    target_tensor = torch.tensor(
        [[ 0.,  0.,  0.],
        [ 10.,  10.,  10.],
        [20., 20., 20.],
        [ 0.,  0.,  0.],
        [ 10.,  10.,  10.],
        [20., 20., 20.]])

    assert torch.equal(new_tensor_0, target_tensor) == True
    assert torch.equal(new_tensor_1, target_tensor) == True

    target_tensor = tensor.tensor([20., 20., 20., 30., 30., 30.])

    assert torch.equal(new_tensor_2, target_tensor) == True

    client.shut_down()

if __name__ == '__main__':
    pid = os.fork()
    if pid == 0:
        start_server()
    else:
        time.sleep(2) # wait server start
        start_client()
