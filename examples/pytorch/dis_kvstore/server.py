import os
import argparse
import time

import dgl
from dgl.contrib import KVServer

import torch as th

g2l = []
g2l.append(th.tensor([0,1,0,0,0,0,0,0]))
g2l.append(th.tensor([0,0,0,1,0,0,0,0]))
g2l.append(th.tensor([0,0,0,0,0,1,0,0]))
g2l.append(th.tensor([0,0,0,0,0,0,0,1]))

data = []
data.append(th.tensor([[4.,4.,4.],[4.,4.,4.]]))
data.append(th.tensor([[3.,3.,3.],[3.,3.,3.]]))
data.append(th.tensor([[2.,2.,2.],[2.,2.,2.]]))
data.append(th.tensor([[1.,1.,1.],[1.,1.,1.]]))

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--machine_id', type=int, default=0,
                          help='Unique ID of each machine.')
        self.add_argument('--server_id', type=int, default=0,
                          help='Unique ID of each server.')
        self.add_argument('--ip_config', type=str, default='ip_config.txt',
                          help='IP configuration file of kvstore.')
        self.add_argument('--backup_count', type=int, default=2,
                          help='Count of backup server.')

def start_server(args):
    """Start kvstore service
    """
    server_namebook = dgl.contrib.read_ip_config(filename=args.ip_config)

    my_server = KVServer(server_id=args.server_id, server_addr=server_namebook[args.server_id], num_client=20)

    if args.server_id % args.backup_count == 0:
        my_server.set_global2local(name='entity_embed', global2local=g2l[args.machine_id])
        my_server.init_data(name='entity_embed', data_tensor=data[args.machine_id])
    else:
        time.sleep(3)
        my_server.set_global2local(name='entity_embed', global2local=None, data_shape=tuple((8,)))
        my_server.init_data(name='entity_embed', data_tensor=None, data_shape=tuple((2,3)))

    my_server.start()
    

if __name__ == '__main__':
    args = ArgParser().parse_args()
    start_server(args)