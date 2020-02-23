import os
import argparse
import time

import dgl
from dgl.contrib import KVClient

import mxnet as mx

partition = mx.nd.array([0,0,1,1,2,2,3,3], dtype='int64')

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


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--ip_config', type=str, default='ip_config.txt',
                          help='IP configuration file of kvstore.')
        self.add_argument('--num_worker', type=int, default=2,
                          help='Number of worker (client nodes) on single-machine.')


def start_client(args):
    """Start client
    """
    server_namebook = dgl.contrib.read_ip_config(filename=args.ip_config)

    my_client = KVClient(server_namebook=server_namebook)

    my_client.connect()

    if my_client.get_id() % args.num_worker == 0:
        my_client.set_partition_book(name='entity_embed', partition_book=partition)
    else:
        my_client.set_partition_book(name='entity_embed')

    my_client.print()

    my_client.barrier()

    print("send request...")

    for i in range(100):
        for i in range(4):
            my_client.push(name='entity_embed', id_tensor=ID[i], data_tensor=DATA[i])

    my_client.barrier()

    if my_client.get_id() % args.num_worker == 0:
        res = my_client.pull(name='entity_embed', id_tensor=mx.nd.array([0,1,2,3,4,5,6,7], dtype='int64'))
        print(res)

    my_client.barrier()

    for i in range(100):
        my_client.push(name='entity_embed', id_tensor=ID[my_client.get_machine_id()], data_tensor=mx.nd.array([[0.,0.,0.],[0.,0.,0.]]))

    my_client.barrier()

    if my_client.get_id() % args.num_worker == 0:
        res = my_client.pull(name='entity_embed', id_tensor=mx.nd.array([0,1,2,3,4,5,6,7], dtype='int64'))
        print(res)

    if my_client.get_id() == 0:
        my_client.shut_down()


if __name__ == '__main__':
    args = ArgParser().parse_args()
    start_client(args)