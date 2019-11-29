import dgl
import argparse
import torch as th
import os
import time

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

server_namebook, client_namebook = dgl.contrib.ReadNetworkConfigure('config.txt')

def start_server(server_id, data_tensor, global_to_local):
    server = dgl.contrib.KVServer(
        server_id=server_id, 
        client_namebook=client_namebook, 
        global_to_local=global_to_local,
        server_addr=server_namebook[server_id])

    server.init_data(name='embed', data_tensor=data_tensor)

    server.start()

def start_client(client_id, data_tensor, global_to_local):
    time.sleep(2) # wait server start

    client = dgl.contrib.KVClient(
        client_id=client_id, 
        local_server_id=client_id, 
        server_namebook=server_namebook, 
        client_addr=client_namebook[client_id],
        partition_book=partition_book,
        global_to_local=global_to_local)

    client.init_local_data(name='embed', data_tensor=data_tensor)

    client.connect()

    if client_id == 0:
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
        print("------result_0------")
        print(result_0)

        ID = th.tensor([6,4,2,0])
        result_1 = client.pull(name='embed', id_tensor=ID)
        print("------result_1------")
        print(result_1)

        ID = th.tensor([7,5,3,1])
        result_2 = client.pull(name='embed', id_tensor=ID)
        print("------result_2------")
        print(result_2)

        ID = th.tensor([3,3,4,4])
        result_3 = client.pull(name='embed', id_tensor=ID)
        print("------result_3------")
        print(result_3)

        ID = th.tensor([0,1])
        result_4 = client.pull(name='embed', id_tensor=ID)
        print("------result_4------")
        print(result_4)

        ID = th.tensor([2,3])
        result_5 = client.pull(name='embed', id_tensor=ID)
        print("------result_5------")
        print(result_5)

        ID = th.tensor([4,5])
        result_6 = client.pull(name='embed', id_tensor=ID)
        print("------result_6------")
        print(result_6)

        ID = th.tensor([6,7])
        result_7 = client.pull(name='embed', id_tensor=ID)
        print("------result_7------")
        print(result_7)

        client.shut_down()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kvstore')
    parser.add_argument("--id", type=int, default=0, help="node ID")
    args = parser.parse_args()

    data_tensor = data_list[args.id]
    data_tensor.share_memory_()

    pid = os.fork()
    if pid == 0:
        start_server(args.id, data_tensor, global_to_local[args.id])
    else:
        start_client(args.id, data_tensor, global_to_local[args.id])

