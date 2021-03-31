import dgl
import argparse
import mxnet as mx
import time
import backend as F

from multiprocessing import Process

ID = []
ID.append(mx.nd.array([0,1], dtype='int64'))
ID.append(mx.nd.array([2,3], dtype='int64'))
ID.append(mx.nd.array([4,5], dtype='int64'))
ID.append(mx.nd.array([6,7], dtype='int64'))

DATA = []
DATA.append(mx.nd.array([[1.,1.,1.,],[1.,1.,1.,]]))
DATA.append(mx.nd.array([[2.,2.,2.,],[2.,2.,2.,]]))
DATA.append(mx.nd.array([[3.,3.,3.,],[3.,3.,3.,]]))
DATA.append(mx.nd.array([[4.,4.,4.,],[4.,4.,4.,]]))

edata_partition_book = {'edata':mx.nd.array([0,0,1,1,2,2,3,3], dtype='int64')}
ndata_partition_book = {'ndata':mx.nd.array([0,0,1,1,2,2,3,3], dtype='int64')}

ndata_g2l = []
edata_g2l = []

ndata_g2l.append({'ndata':mx.nd.array([0,1,0,0,0,0,0,0], dtype='int64')})
ndata_g2l.append({'ndata':mx.nd.array([0,0,0,1,0,0,0,0], dtype='int64')})
ndata_g2l.append({'ndata':mx.nd.array([0,0,0,0,0,1,0,0], dtype='int64')})
ndata_g2l.append({'ndata':mx.nd.array([0,0,0,0,0,0,0,1], dtype='int64')})

edata_g2l.append({'edata':mx.nd.array([0,1,0,0,0,0,0,0], dtype='int64')})
edata_g2l.append({'edata':mx.nd.array([0,0,0,1,0,0,0,0], dtype='int64')})
edata_g2l.append({'edata':mx.nd.array([0,0,0,0,0,1,0,0], dtype='int64')})
edata_g2l.append({'edata':mx.nd.array([0,0,0,0,0,0,0,1], dtype='int64')})

def start_client(flag):
    time.sleep(3)

    client = dgl.contrib.start_client(ip_config='ip_config.txt', 
                                      ndata_partition_book=ndata_partition_book, 
                                      edata_partition_book=edata_partition_book,
                                      close_shared_mem=flag)

    client.push(name='edata', id_tensor=ID[client.get_id()], data_tensor=DATA[client.get_id()])
    client.push(name='ndata', id_tensor=ID[client.get_id()], data_tensor=DATA[client.get_id()])

    client.barrier()

    tensor_edata = client.pull(name='edata', id_tensor=mx.nd.array([0,1,2,3,4,5,6,7], dtype='int64'))
    tensor_ndata = client.pull(name='ndata', id_tensor=mx.nd.array([0,1,2,3,4,5,6,7], dtype='int64'))


    target_tensor = mx.nd.array([[1., 1., 1.],
                               [1., 1., 1.],
                               [2., 2., 2.],
                               [2., 2., 2.],
                               [3., 3., 3.],
                               [3., 3., 3.],
                               [4., 4., 4.],
                               [4., 4., 4.]])

    assert F.array_equal(tensor_edata, target_tensor)

    assert F.array_equal(tensor_ndata, target_tensor)

    client.barrier()

    if client.get_id() == 0:
        client.shut_down()

def start_server(server_id, num_client):
    
    dgl.contrib.start_server(
        server_id=server_id,
        ip_config='ip_config.txt',
        num_client=num_client,
        ndata={'ndata':mx.nd.array([[0.,0.,0.],[0.,0.,0.]])},
        edata={'edata':mx.nd.array([[0.,0.,0.],[0.,0.,0.]])},
        ndata_g2l=ndata_g2l[server_id],
        edata_g2l=edata_g2l[server_id])

if __name__ == '__main__':

    # server process
    p0 = Process(target=start_server, args=(0, 4))
    p1 = Process(target=start_server, args=(1, 4))
    p2 = Process(target=start_server, args=(2, 4))
    p3 = Process(target=start_server, args=(3, 4))

    # client process
    p4 = Process(target=start_client, args=(True,))
    p5 = Process(target=start_client, args=(True,))
    p6 = Process(target=start_client, args=(False,))
    p7 = Process(target=start_client, args=(False,))


    # start server process
    p0.start()
    p1.start()
    p2.start()
    p3.start()

    # start client process
    p4.start()
    p5.start()
    p6.start()
    p7.start()


    p0.join()
    p1.join()
    p2.join()
    p3.join()

    p4.join()
    p5.join()
    p6.join()
    p7.join()