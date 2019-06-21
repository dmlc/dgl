import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tqdm
from rec.model.pinsage import PinSage
from rec.utils import cuda
from rec.comm.sender import NodeFlowSender
from dgl import DGLGraph
from dgl.contrib.sampling import NeighborSampler
from validation import *

import argparse
import pickle
import os
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--n-negs', type=int, default=1)
parser.add_argument('--cache', type=str, default='/tmp/dataset.pkl',
                    help='File to cache the postprocessed dataset object')
parser.add_argument('--raw-dataset-path', type=str, default='/efs/quagan/movielens')
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

n_negs = args.n_negs

n_users = len(ml.user_ids)
n_items = len(ml.product_ids)

# Find negative examples for each positive example
# Note that this function only pick negative examples for products; it doesn't
# allow picking for users.
def find_negs(g_train, dst, ml, n_negs):
    dst_neg = []
    for i, d in enumerate(dst):
        assert len(ml.user_ids) <= d < len(ml.user_ids) + len(ml.product_ids)
        dst_neg.append(np.random.randint(
            len(ml.user_ids), len(ml.user_ids) + len(ml.product_ids), n_negs))
    dst_neg = torch.LongTensor(dst_neg)
    return dst_neg

sender = NodeFlowSender(args.host, args.port)
g.readonly()
g_train_edges = g.filter_edges(lambda edges: edges.data['train'])
g_train = g.edge_subgraph(g_train_edges, True)

for epoch in range(500):
    edge_shuffled = torch.LongTensor(sender.recv())
    src, dst = g_train.find_edges(edge_shuffled)
    # Chop the users, items and negative items into batches and reorganize
    # them into seed nodes.
    # Note that the batch size of DGL sampler here is (2 + n_negs) times our batch size,
    # since the sampler is handling (2 + n_negs) nodes per training example.
    edge_batches = edge_shuffled.split(batch_size)
    src_batches = src.split(batch_size)
    dst_batches = dst.split(batch_size)
    if n_negs > 0:
        dst_neg = find_negs(g_train, dst, ml, n_negs)
        dst_neg = dst_neg.flatten()
        dst_neg_batches = dst_neg.split(batch_size * n_negs)

    seed_nodes = []
    for i in range(len(src_batches)):
        seed_nodes.append(src_batches[i])
        seed_nodes.append(dst_batches[i])
        if n_negs > 0:
            seed_nodes.append(dst_neg_batches[i])
    seed_nodes = torch.cat(seed_nodes)

    sampler = NeighborSampler(
            g_train,
            batch_size * (2 + n_negs),
            5,
            n_layers,
            seed_nodes=seed_nodes,
            prefetch=True,
            add_self_loop=True,
            num_workers=20)
    sampler_iter = iter(sampler)

    with tqdm.tqdm(sampler) as tq:
        for batch_id, nodeflow in enumerate(tq):
            edges = edge_batches[batch_id]
            src = src_batches[batch_id]
            dst = dst_batches[batch_id]
            if n_negs > 0:
                dst_neg = dst_neg_batches[batch_id]

            sender.send(
                    nodeflow,
                    (edges.numpy(),
                     src.numpy(),
                     dst.numpy(),
                     dst_neg.numpy() if n_negs > 0 else None))
    sender.complete()
