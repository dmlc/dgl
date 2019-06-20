import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tqdm
from rec.utils import cuda
from rec.comm.sender import NodeFlowSender
from dgl import DGLGraph
from dgl.contrib.sampling import NeighborSampler

import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--n-negs', type=int, default=1)
parser.add_argument('--cache', type=str, default='/tmp/dataset.pkl',
parser.add_argument('--raw-dataset-path', type=str, default='/efs/quagan/movielens')
                    help='File to cache the postprocessed dataset object')
parser.add_argument('--dataset', type=str, default='movielens')
parser.add_argument('--host', type=str, default='localhost')
parser.add_argument('--port', type=int, default=5902)
parser.add_argument('--batch-size', type=int, default=32)
args = parser.parse_args()

print(args)

ml = load_data(args)

g = ml.g
n_layers = args.layers
batch_size = args.batch_size

n_users = len(ml.user_ids)
n_items = len(ml.product_ids)

g.readonly()

sender = NodeFlowSender(args.host, args.port)

for epoch in range(500):
    seeds = torch.LongTensor(sender.recv()) + n_users
    sampler = NeighborSampler(
            g_prior,
            batch_size,
            5,
            n_layers,
            seed_nodes=seeds,
            prefetch=True,
            add_self_loop=True,
            num_workers=20)

    with tqdm.tqdm(sampler) as tq:
        for i, nodeflow in enumerate(tq):
            sender.send(
                    nodeflow,
                    seeds[i * batch_size:(i + 1) * batch_size].numpy() - n_users)
    print('Completing')
    sender.complete()
