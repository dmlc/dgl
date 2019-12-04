import os
import time
import argparse

import dgl
import torch as th
import torch.multiprocessing as mp

NUM_PROC = 8

# We use a simple range partition in this demo.
# 8 rows of data, and each machine has 2 rows of data.
partition_book = [0,0,1,1,2,2,3,3]

# For real tasks, the global_to_local data could be read from file.
global_to_local = []
global_to_local.append([0,1,0,0,0,0,0,0])
global_to_local.append([0,0,0,1,0,0,0,0])
global_to_local.append([0,0,0,0,0,1,0,0])
global_to_local.append([0,0,0,0,0,0,0,1])

data_list = []
data_list.append(th.tensor([[1.,1.],[2.,2.]]))
data_list.append(th.tensor([[3.,3.],[4.,4.]]))
data_list.append(th.tensor([[5.,5.],[6.,6.]]))
data_list.append(th.tensor([[7.,7.],[8.,8.]]))

server_namebook = { 0:'127.0.0.1:50055', 1:'127.0.0.1:50056', 2:'127.0.0.1:50057', 3:'127.0.0.1:50058' }
client_namebook = { 0:'127.0.0.1:50051', 1:'127.0.0.1:50052', 2:'127.0.0.1:50053', 3:'127.0.0.1:50054' }

def start_server(server_id, data_tensor, global_to_local):
    server = dgl.contrib.KVServer(
        server_id=server_id, 
        client_namebook=client_namebook, 
        server_addr=server_namebook[server_id])

    server.set_global_to_local(name='embed', global_to_local=global_to_local)

    server.init_data(name='embed', data_tensor=data_tensor)

    server.start()

def start_client(client_id, data_tensor, global_to_local):
    time.sleep(2) # wait server start

    client = dgl.contrib.KVClient(
        client_id=client_id, 
        local_server_id=client_id, 
        server_namebook=server_namebook, 
        client_addr=client_namebook[client_id])

    client.set_partition_book(name='embed', partition_book=partition_book)

    client.set_global_to_local(name='embed', global_to_local=global_to_local)

    client.init_data(name='embed', data_tensor=data_tensor)

    client.connect()

    # push
    ID = th.tensor([0,1,2,3,4,5,6,7])
    data = th.tensor([[1.,1.],[1.,1.],[1.,1.],[1.,1.],[1.,1.],[1.,1.],[1.,1.],[1.,1.]])
    client.push(name='embed', id_tensor=ID, data_tensor=data)

    # push 
    ID = th.tensor([0,2,4,6])
    data = th.tensor([[1.,1.],[1.,1.],[1.,1.],[1.,1.]])
    client.push(name='embed', id_tensor=ID, data_tensor=data)

    # push
    ID = th.tensor([1,3,5,7])
    data = th.tensor([[1.,1.],[1.,1.],[1.,1.],[1.,1.]])
    client.push(name='embed', id_tensor=ID, data_tensor=data)

    # push
    ID = th.tensor([2,4,3,1,7,5,0,6])
    data = th.tensor([[1.,1.],[1.,1.],[1.,1.],[1.,1.],[1.,1.],[1.,1.],[1.,1.],[1.,1.]])
    client.push(name='embed', id_tensor=ID, data_tensor=data)

    # push
    ID = th.tensor([0,1])
    data = th.tensor([[1.,1.],[1.,1.]])
    client.push(name='embed', id_tensor=ID, data_tensor=data)

    # push
    ID = th.tensor([2,3])
    data = th.tensor([[1.,1.],[1.,1.]])
    client.push(name='embed', id_tensor=ID, data_tensor=data)

    # push
    ID = th.tensor([4,5])
    data = th.tensor([[1.,1.],[1.,1.]])
    client.push(name='embed', id_tensor=ID, data_tensor=data)

    # push
    ID = th.tensor([6,7])
    data = th.tensor([[1.,1.],[1.,1.]])
    client.push(name='embed', id_tensor=ID, data_tensor=data)


    # wait all push() finish
    client.barrier()

    if client_id == 0:
        ID = th.tensor([0,1,2,3,4,5,6,7])
        result_0 = client.pull(name='embed', id_tensor=ID)
        target_tensor = th.tensor(
          [[17., 17.],
           [18., 18.],
           [19., 19.],
           [20., 20.],
           [21., 21.],
           [22., 22.],
           [23., 23.],
           [24., 24.]])
        assert th.equal(result_0, target_tensor) == True

        ID = th.tensor([6,4,2,0])
        result_1 = client.pull(name='embed', id_tensor=ID)
        target_tensor = th.tensor(
          [[23., 23.],
           [21., 21.],
           [19., 19.],
           [17., 17.]])
        assert th.equal(result_1, target_tensor) == True


        ID = th.tensor([7,5,3,1])
        result_2 = client.pull(name='embed', id_tensor=ID)
        target_tensor = th.tensor(
          [[24., 24.],
           [22., 22.],
           [20., 20.],
           [18., 18.]])
        assert th.equal(result_2, target_tensor) == True


        ID = th.tensor([3,3,4,4])
        result_3 = client.pull(name='embed', id_tensor=ID)
        target_tensor = th.tensor(
          [[20., 20.],
           [20., 20.],
           [21., 21.],
           [21., 21.]])
        assert th.equal(result_3, target_tensor) == True


        ID = th.tensor([0,1])
        result_4 = client.pull(name='embed', id_tensor=ID)
        target_tensor = th.tensor(
          [[17., 17.],
           [18., 18.]])
        assert th.equal(result_4, target_tensor) == True

        ID = th.tensor([2,3])
        result_5 = client.pull(name='embed', id_tensor=ID)
        target_tensor = th.tensor(
          [[19., 19.],
           [20., 20.]])
        assert th.equal(result_5, target_tensor) == True

        ID = th.tensor([4,5])
        result_6 = client.pull(name='embed', id_tensor=ID)
        target_tensor = th.tensor(
          [[21., 21.],
           [22., 22.]])
        assert th.equal(result_6, target_tensor) == True

        ID = th.tensor([6,7])
        result_7 = client.pull(name='embed', id_tensor=ID)
        target_tensor = th.tensor(
          [[23., 23.],
           [24., 24.]])
        assert th.equal(result_7, target_tensor) == True

        ID = th.tensor([0,0,0,0,0])
        result_8 = client.pull(name='embed', id_tensor=ID)
        target_tensor = th.tensor(
          [[17., 17.],
           [17., 17.],
           [17., 17.],
           [17., 17.],
           [17., 17.]])
        assert th.equal(result_8, target_tensor) == True

        client.shut_down()

if __name__ == '__main__':
    for data in data_list:
        data.share_memory_()

    procs = []
    for i in range(NUM_PROC):
        if i < 4:
            proc = mp.Process(target=start_server, args=(i, data_list[i], global_to_local[i]))
        else:
            proc = mp.Process(target=start_client, args=(i-4, data_list[i-4], global_to_local[i-4]))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()

