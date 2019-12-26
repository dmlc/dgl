# This is a simple MXNet server demo shows how to use DGL distributed kvstore.
import dgl
import argparse
import torch as th
import time

ID = []
ID.append(th.tensor([0,1]))
ID.append(th.tensor([2,3]))
ID.append(th.tensor([4,5]))
ID.append(th.tensor([6,7]))

edata_partition_book = {'edata':th.tensor([0,0,1,1,2,2,3,3])}
ndata_partition_book = {'ndata':th.tensor([0,0,1,1,2,2,3,3])}

def start_client():
    time.sleep(3)

    client = dgl.contrib.start_client(ip_config='ip_config.txt', 
                                      ndata_partition_book=ndata_partition_book, 
                                      edata_partition_book=edata_partition_book)

    client.push(name='edata', id_tensor=ID[client.get_id()], data_tensor=th.tensor([[1.,1.,1.],[1.,1.,1.]]))
    client.push(name='ndata', id_tensor=ID[client.get_id()], data_tensor=th.tensor([[2.,2.,2.],[2.,2.,2.]]))

    client.barrier()

    tensor_edata = client.pull(name='edata', id_tensor=th.tensor([0,1,2,3,4,5,6,7]))
    tensor_ndata = client.pull(name='ndata', id_tensor=th.tensor([0,1,2,3,4,5,6,7]))

    print(tensor_edata)

    client.barrier()

    print(tensor_ndata)

    client.barrier()

    if client.get_id() == 0:
        client.shut_down()

if __name__ == '__main__':

    start_client()