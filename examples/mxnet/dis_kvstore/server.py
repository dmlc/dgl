import os
import argparse
import time

import dgl
from dgl.contrib import KVServer

import mxnet as mx

g2l = []
g2l.append(mx.nd.array([0,1,0,0,0,0,0,0], dtype='int64'))
g2l.append(mx.nd.array([0,0,0,1,0,0,0,0], dtype='int64'))
g2l.append(mx.nd.array([0,0,0,0,0,1,0,0], dtype='int64'))
g2l.append(mx.nd.array([0,0,0,0,0,0,0,1], dtype='int64'))

data = []
data.append(mx.nd.array([[4.,4.,4.],[4.,4.,4.]]))
data.append(mx.nd.array([[3.,3.,3.],[3.,3.,3.]]))
data.append(mx.nd.array([[2.,2.,2.],[2.,2.,2.]]))
data.append(mx.nd.array([[1.,1.,1.],[1.,1.,1.]]))


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--server_id', type=int, default=0,
                          help='Unique ID of each server.')
        self.add_argument('--ip_config', type=str, default='ip_config.txt',
                          help='IP configuration file of kvstore.')
        self.add_argument('--num_client', type=int, default=1,
                          help='Total number of client nodes.')


def start_server(args):
    """Start kvstore service
    """
    server_namebook = dgl.contrib.read_ip_config(filename=args.ip_config)

    my_server = KVServer(server_id=args.server_id, server_namebook=server_namebook, num_client=args.num_client)

    if my_server.get_id() % my_server.get_group_count() == 0: # master server
        my_server.set_global2local(name='entity_embed', global2local=g2l[my_server.get_machine_id()])
        my_server.init_data(name='entity_embed', data_tensor=data[my_server.get_machine_id()])
    else:
        my_server.set_global2local(name='entity_embed')
        my_server.init_data(name='entity_embed')

    my_server.print()

    my_server.start()
    

if __name__ == '__main__':
    args = ArgParser().parse_args()
    start_server(args)