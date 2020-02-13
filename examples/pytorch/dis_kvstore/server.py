import os
import argparse
import time

import dgl
from dgl.contrib import KVServer

import torch as th


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--machine_number', type=int, default=1,
                          help='Total number of machine.')
        self.add_argument('--server_id', type=int, default=0,
                          help='Unique ID of each server.')
        self.add_argument('--ip_config', type=str, default='ip_config.txt',
                          help='IP configuration file of kvstore.')


def start_server(args):
    """Start kvstore service
    """
    server_namebook = dgl.contrib.read_ip_config(filename=args.ip_config)

    my_server = KVServer(server_id=args.server_id, server_addr=server_namebook[args.server_id], num_client=20)

    if args.server_id % args.machine_number == 0:
        my_server.set_global2local(name='entity_embed', global2local=th.tensor([0,1,2]))
        my_server.init_data(name='entity_embed', data_tensor=th.zeros(50000000,200))
    else:
        time.sleep(3)
        my_server.set_global2local(name='entity_embed', global2local=None, shared_server_id=0, data_shape=tuple((3,)))
        my_server.init_data(name='entity_embed', data_tensor=None, shared_server_id=0, data_shape=tuple((50000000,200)))

    my_server.start()
    

if __name__ == '__main__':
    args = ArgParser().parse_args()
    start_server(args)