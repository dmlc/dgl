import os
import argparse
import time

import dgl
from dgl.contrib import KVClient

import torch as th

partition = th.tensor([0,0,1,1,2,2,3,3])

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--machine_id', type=int, default=0,
                          help='Unique ID of each machine.')
        self.add_argument('--ip_config', type=str, default='ip_config.txt',
                          help='IP configuration file of kvstore.')


def start_client(args):
    """Start kvstore service
    """
    server_namebook = dgl.contrib.read_ip_config(filename=args.ip_config)

    my_client = KVClient(server_namebook)

    my_client.set_partition_book(name='entity_embed', partition_book=partition)

    my_client.connect()
    

if __name__ == '__main__':
    args = ArgParser().parse_args()
    start_client(args)