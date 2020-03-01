import os
import argparse
import time

import dgl
from dgl.contrib import KVServer

import torch as th

from train_pytorch import load_model
from dataloader import get_server_partition_dataset


NUM_THREAD = 1 # Fix the number of threads to 1 on kvstore

class KGEServer(KVServer):
    """User-defined kvstore for DGL-KGE
    """
    def _push_handler(self, name, ID, data, target):
        """Row-Sparse Adagrad updater
        """
        original_name = name[0:-6]
        state_sum = target[original_name+'_state-data-']
        grad_sum = (data * data).mean(1)
        state_sum.index_add_(0, ID, grad_sum)
        std = state_sum[ID]  # _sparse_mask
        std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
        tmp = (-self.clr * data / std_values)
        target[name].index_add_(0, ID, tmp)


    def set_clr(self, learning_rate):
        """Set learning rate for Row-Sparse Adagrad updater
        """
        self.clr = learning_rate


# Note: Most of the args are unnecessary for KVStore, will remove them later
class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--model_name', default='TransE',
                          choices=['TransE', 'TransE_l1', 'TransE_l2', 'TransR',
                                   'RESCAL', 'DistMult', 'ComplEx', 'RotatE'],
                          help='model to use')
        self.add_argument('--data_path', type=str, default='../data',
                          help='root path of all dataset')
        self.add_argument('--dataset', type=str, default='FB15k',
                          help='dataset name, under data_path')
        self.add_argument('--format', type=str, default='1',
                          help='the format of the dataset.')

        self.add_argument('--hidden_dim', type=int, default=256,
                          help='hidden dim used by relation and entity')
        self.add_argument('--lr', type=float, default=0.0001,
                          help='learning rate')
        self.add_argument('-g', '--gamma', type=float, default=12.0,
                          help='margin value')

        self.add_argument('--gpu', type=int, default=[-1], nargs='+',
                          help='a list of active gpu ids, e.g. 0')
        self.add_argument('--mix_cpu_gpu', action='store_true',
                          help='mix CPU and GPU training')
        self.add_argument('-de', '--double_ent', action='store_true',
                          help='double entitiy dim for complex number')
        self.add_argument('-dr', '--double_rel', action='store_true',
                          help='double relation dim for complex number')
        self.add_argument('--seed', type=int, default=0,
                          help='set random seed for reproducibility')

        self.add_argument('--rel_part', action='store_true',
                          help='enable relation partitioning')
        self.add_argument('--soft_rel_part', action='store_true',
                          help='enable soft relation partition')
        self.add_argument('--nomp_thread_per_process', type=int, default=-1,
                          help='num of omp threads used per process in multi-process training')
        self.add_argument('--async_update', action='store_true',
                          help='allow async_update on node embedding')
        self.add_argument('--strict_rel_part', action='store_true',
                          help='Strict relation partition')

        self.add_argument('--server_id', type=int, default=0,
                          help='Unique ID of each server')
        self.add_argument('--ip_config', type=str, default='ip_config.txt',
                          help='IP configuration file of kvstore')
        self.add_argument('--total_client', type=int, default=1,
                          help='Total number of client worker nodes')


def get_server_data(args, machine_id):
   """Get data from data_path/dataset/part_machine_id

      Return: glocal2local, 
              entity_emb, 
              entity_state, 
              relation_emb, 
              relation_emb_state
   """
   g2l, dataset = get_server_partition_dataset(
    args.data_path, 
    args.dataset, 
    args.format, 
    machine_id)

   # Note that the dataset doesn't ccontain the triple
   print('n_entities: ' + str(dataset.n_entities))
   print('n_relations: ' + str(dataset.n_relations))

   model = load_model(None, args, dataset.n_entities, dataset.n_relations)

   return g2l, model.entity_emb.emb, model.entity_emb.state_sum, model.relation_emb.emb, model.relation_emb.state_sum


def start_server(args):
    """Start kvstore service
    """
    th.set_num_threads(NUM_THREAD)

    server_namebook = dgl.contrib.read_ip_config(filename=args.ip_config)

    my_server = KGEServer(server_id=args.server_id, 
                          server_namebook=server_namebook, 
                          num_client=args.total_client)

    my_server.set_clr(args.lr)

    if my_server.get_id() % my_server.get_group_count() == 0: # master server
        g2l, entity_emb, entity_emb_state, relation_emb, relation_emb_state = get_server_data(args, my_server.get_machine_id())
        my_server.set_global2local(name='entity_emb', global2local=g2l)
        my_server.init_data(name='relation_emb', data_tensor=relation_emb)
        my_server.init_data(name='relation_emb_state', data_tensor=relation_emb_state)
        my_server.init_data(name='entity_emb', data_tensor=entity_emb)
        my_server.init_data(name='entity_emb_state', data_tensor=entity_emb_state)
    else: # backup server
        my_server.set_global2local(name='entity_emb')
        my_server.init_data(name='relation_emb')
        my_server.init_data(name='relation_emb_state')
        my_server.init_data(name='entity_emb')
        my_server.init_data(name='entity_emb_state')

    print('KVServer %d listen for requests ...' % my_server.get_id())

    my_server.start()
    

if __name__ == '__main__':
    args = ArgParser().parse_args()
    start_server(args)