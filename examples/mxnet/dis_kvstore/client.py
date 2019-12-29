# This is a simple MXNet server demo shows how to use DGL distributed kvstore.
import dgl
import argparse
import mxnet as mx
import time

ID = []
ID.append(mx.nd.array([0,1], dtype='int64'))
ID.append(mx.nd.array([2,3], dtype='int64'))
ID.append(mx.nd.array([4,5], dtype='int64'))
ID.append(mx.nd.array([6,7], dtype='int64'))

DATA = []
DATA.append(mx.nd.array([[1.,1.,1.,],[1.,1.,1.,]], dtype='int64'))
DATA.append(mx.nd.array([[2.,2.,2.,],[2.,2.,2.,]], dtype='int64'))
DATA.append(mx.nd.array([[3.,3.,3.,],[3.,3.,3.,]], dtype='int64'))
DATA.append(mx.nd.array([[4.,4.,4.,],[4.,4.,4.,]], dtype='int64'))

edata_partition_book = {'edata':mx.nd.array([0,0,1,1,2,2,3,3], dtype='int64')}
ndata_partition_book = {'ndata':mx.nd.array([0,0,1,1,2,2,3,3], dtype='int64')}

def start_client():
    time.sleep(3)

    client = dgl.contrib.start_client(ip_config='ip_config.txt', 
                                      ndata_partition_book=ndata_partition_book, 
                                      edata_partition_book=edata_partition_book)


    tensor_edata = client.pull(name='edata', id_tensor=mx.nd.array([0,1,2,3,4,5,6,7], dtype='int64'))
    tensor_ndata = client.pull(name='ndata', id_tensor=mx.nd.array([0,1,2,3,4,5,6,7], dtype='int64'))

    print(tensor_edata)
    client.barrier()

    print(tensor_ndata)
    client.barrier()

    client.push(name='edata', id_tensor=ID[client.get_id()], data_tensor=DATA[client.get_id()])
    client.push(name='ndata', id_tensor=ID[client.get_id()], data_tensor=DATA[client.get_id()])

    client.barrier()

    tensor_edata = client.pull(name='edata', id_tensor=mx.nd.array([0,1,2,3,4,5,6,7], dtype='int64'))
    tensor_ndata = client.pull(name='ndata', id_tensor=mx.nd.array([0,1,2,3,4,5,6,7], dtype='int64'))

    print(tensor_edata)
    client.barrier()

    print(tensor_ndata)
    client.barrier()

    if client.get_id() == 0:
        client.shut_down()

if __name__ == '__main__':

    start_client()