import os
import argparse
import time
import logging

import socket
if os.name != 'nt':
    import fcntl
    import struct

import torch.multiprocessing as mp
from train_pytorch import load_model, dist_train_test

from train import get_logger
from dataloader import TrainDataset, NewBidirectionalOneShotIterator
from dataloader import get_dataset, get_partition_dataset

import dgl
import dgl.backend as F

NUM_THREAD = 1 # Fix the number of threads to 1 on kvclient
NUM_WORKER = 1 # Fix the number of worker for sampler to 1

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
        self.add_argument('--save_path', type=str, default='ckpts',
                          help='place to save models and logs')
        self.add_argument('--save_emb', type=str, default=None,
                          help='save the embeddings in the specific location.')

        self.add_argument('--max_step', type=int, default=80000,
                          help='train xx steps')
        self.add_argument('--warm_up_step', type=int, default=None,
                          help='for learning rate decay')
        self.add_argument('--batch_size', type=int, default=1024,
                          help='batch size')
        self.add_argument('--batch_size_eval', type=int, default=8,
                          help='batch size used for eval and test')
        self.add_argument('--neg_sample_size', type=int, default=128,
                          help='negative sampling size')
        self.add_argument('--neg_chunk_size', type=int, default=-1,
                          help='chunk size of the negative edges.')
        self.add_argument('--neg_deg_sample', action='store_true',
                          help='negative sample proportional to vertex degree in the training')
        self.add_argument('--neg_deg_sample_eval', action='store_true',
                          help='negative sampling proportional to vertex degree in the evaluation')
        self.add_argument('--neg_sample_size_valid', type=int, default=1000,
                          help='negative sampling size for validation')
        self.add_argument('--neg_chunk_size_valid', type=int, default=-1,
                          help='chunk size of the negative edges.')
        self.add_argument('--neg_sample_size_test', type=int, default=-1,
                          help='negative sampling size for testing')
        self.add_argument('--neg_chunk_size_test', type=int, default=-1,
                          help='chunk size of the negative edges.')
        self.add_argument('--hidden_dim', type=int, default=256,
                          help='hidden dim used by relation and entity')
        self.add_argument('--lr', type=float, default=0.0001,
                          help='learning rate')
        self.add_argument('-g', '--gamma', type=float, default=12.0,
                          help='margin value')
        self.add_argument('--eval_percent', type=float, default=1,
                          help='sample some percentage for evaluation.')
        self.add_argument('--no_eval_filter', action='store_true',
                          help='do not filter positive edges among negative edges for evaluation')

        self.add_argument('--gpu', type=int, default=[-1], nargs='+', 
                          help='a list of active gpu ids, e.g. 0 1 2 4')
        self.add_argument('--mix_cpu_gpu', action='store_true',
                          help='mix CPU and GPU training')
        self.add_argument('-de', '--double_ent', action='store_true',
                          help='double entitiy dim for complex number')
        self.add_argument('-dr', '--double_rel', action='store_true',
                          help='double relation dim for complex number')
        self.add_argument('--seed', type=int, default=0,
                          help='set random seed fro reproducibility')
        self.add_argument('-log', '--log_interval', type=int, default=1000,
                          help='do evaluation after every x steps')
        self.add_argument('--eval_interval', type=int, default=10000,
                          help='do evaluation after every x steps')
        self.add_argument('-adv', '--neg_adversarial_sampling', action='store_true',
                          help='if use negative adversarial sampling')
        self.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)

        self.add_argument('--valid', action='store_true',
                          help='if valid a model')
        self.add_argument('--test', action='store_true',
                          help='if test a model')
        self.add_argument('-rc', '--regularization_coef', type=float, default=0.000002,
                          help='set value > 0.0 if regularization is used')
        self.add_argument('-rn', '--regularization_norm', type=int, default=3,
                          help='norm used in regularization')
        self.add_argument('--num_worker', type=int, default=32,
                          help='number of workers used for loading data')
        self.add_argument('--non_uni_weight', action='store_true',
                          help='if use uniform weight when computing loss')
        self.add_argument('--init_step', type=int, default=0,
                          help='DONT SET MANUALLY, used for resume')
        self.add_argument('--step', type=int, default=0,
                          help='DONT SET MANUALLY, track current step')
        self.add_argument('--pickle_graph', action='store_true',
                          help='pickle built graph, building a huge graph is slow.')
        self.add_argument('--num_proc', type=int, default=1,
                          help='number of process used')
        self.add_argument('--rel_part', action='store_true',
                          help='enable relation partitioning')
        self.add_argument('--soft_rel_part', action='store_true',
                          help='enable soft relation partition')
        self.add_argument('--nomp_thread_per_process', type=int, default=-1,
                          help='num of omp threads used per process in multi-process training')
        self.add_argument('--async_update', action='store_true',
                          help='allow async_update on node embedding')
        self.add_argument('--force_sync_interval', type=int, default=-1,
                          help='We force a synchronization between processes every x steps')
        self.add_argument('--strict_rel_part', action='store_true',
                          help='Strict relation partition')

        self.add_argument('--machine_id', type=int, default=0,
                          help='Unique ID of current machine.')
        self.add_argument('--total_machine', type=int, default=1,
                          help='Total number of machine.')
        self.add_argument('--ip_config', type=str, default='ip_config.txt',
                          help='IP configuration file of kvstore')
        self.add_argument('--num_client', type=int, default=1,
                          help='Number of client on each machine.')


def get_long_tail_partition(n_relations, n_machine):
    """Relation types has a long tail distribution for many dataset.
       So we need to average shuffle the data before we partition it.
    """
    assert n_relations > 0, 'n_relations must be a positive number.'
    assert n_machine > 0, 'n_machine must be a positive number.'

    partition_book = [0] * n_relations

    part_id = 0
    for i in range(n_relations):
        partition_book[i] = part_id
        part_id += 1
        if part_id == n_machine:
          part_id = 0

    return partition_book 


def local_ip4_addr_list():
    """Return a set of IPv4 address
    """
    nic = set()

    for ix in socket.if_nameindex():
        name = ix[1]
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip = socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack('256s', name[:15].encode("UTF-8")))[20:24])
        nic.add(ip)

    return nic


def get_local_machine_id(server_namebook):
    """Get machine ID via server_namebook
    """
    assert len(server_namebook) > 0, 'server_namebook cannot be empty.'

    res = 0
    for ID, data in server_namebook.items():
        machine_id = data[0]
        ip = data[1]
        if ip in local_ip4_addr_list():
            res = machine_id
            break

    return res


def start_worker(args, logger):
    """Start kvclient for training
    """
    train_time_start = time.time()

    server_namebook = dgl.contrib.read_ip_config(filename=args.ip_config)

    args.machine_id = get_local_machine_id(server_namebook)

    dataset, entity_partition_book, local2global = get_partition_dataset(
        args.data_path,
        args.dataset,
        args.format,
        args.machine_id)

    n_entities = dataset.n_entities
    n_relations = dataset.n_relations

    print('Partition %d n_entities: %d' % (args.machine_id, n_entities))
    print("Partition %d n_relations: %d" % (args.machine_id, n_relations))

    entity_partition_book = F.tensor(entity_partition_book)
    relation_partition_book = get_long_tail_partition(dataset.n_relations, args.total_machine)
    relation_partition_book = F.tensor(relation_partition_book)
    local2global = F.tensor(local2global)

    relation_partition_book.share_memory_()
    entity_partition_book.share_memory_()
    local2global.share_memory_()

    model = load_model(logger, args, n_entities, n_relations)
    model.share_memory()

    # When we generate a batch of negative edges from a set of positive edges,
    # we first divide the positive edges into chunks and corrupt the edges in a chunk
    # together. By default, the chunk size is equal to the negative sample size.
    # Usually, this works well. But we also allow users to specify the chunk size themselves.
    if args.neg_chunk_size < 0:
        args.neg_chunk_size = args.neg_sample_size

    num_workers = NUM_WORKER
    train_data = TrainDataset(dataset, args, ranks=args.num_client)

    train_samplers = []
    for i in range(args.num_client):
        train_sampler_head = train_data.create_sampler(args.batch_size,
                                                       args.neg_sample_size,
                                                       args.neg_chunk_size,
                                                       mode='head',
                                                       num_workers=num_workers,
                                                       shuffle=True,
                                                       exclude_positive=False,
                                                       rank=i)
        train_sampler_tail = train_data.create_sampler(args.batch_size,
                                                       args.neg_sample_size,
                                                       args.neg_chunk_size,
                                                       mode='tail',
                                                       num_workers=num_workers,
                                                       shuffle=True,
                                                       exclude_positive=False,
                                                       rank=i)
        train_samplers.append(NewBidirectionalOneShotIterator(train_sampler_head, train_sampler_tail,
                                                              args.neg_chunk_size, args.neg_sample_size,
                                                              True, n_entities))

    dataset = None

    print('Total data loading time {:.3f} seconds'.format(time.time() - train_time_start))

    rel_parts = train_data.rel_parts if args.strict_rel_part or args.soft_rel_part else None
    cross_rels = train_data.cross_rels if args.soft_rel_part else None

    args.num_thread = NUM_THREAD

    procs = []
    barrier = mp.Barrier(args.num_client)
    for i in range(args.num_client):
        proc = mp.Process(target=dist_train_test, args=(args,
                                                        model,
                                                        train_samplers[i],
                                                        entity_partition_book,
                                                        relation_partition_book,
                                                        local2global,
                                                        i,
                                                        rel_parts,
                                                        cross_rels,
                                                        barrier))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()


if __name__ == '__main__':
    args = ArgParser().parse_args()
    logger = get_logger(args)
    start_worker(args, logger)