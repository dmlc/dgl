import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from rec.model.pinsage import PinSage
from rec.datasets.movielens import MovieLens
from rec.utils import cuda

import pickle
import os

cache_file = 'ml.pkl'

if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        ml = pickle.load(f)
else:
    ml = MovieLens('./ml-1m')
    with open(cache_file, 'wb') as f:
        pickle.dump(ml, f)

g = ml.g
neighbors = ml.user_neighbors + ml.movie_neighbors

n_hidden = 100
n_layers = 3
batch_size = 32

# Use the prior graph to train on user-product pairs in the training set.
# Validate on validation set.
# Note that each user-product pair is counted twice, but I think it is OK
# since we can treat product negative sampling and user negative sampling
# ubiquitously.
g_prior_edges = g.filter_edges(lambda edges: edges.data['prior'])
g_train_edges = g.filter_edges(lambda edges: edges.data['train'])
g_valid_edges = g.filter_edges(lambda edges: edges.data['valid'])
g_prior = g.edge_subgraph(g_prior_edges)
g_prior_nid = g_prior.parent_nid
g_prior_nid_np = g_prior_nid.numpy()

model = cuda(PinSage(g_prior, [n_hidden] * n_layers, 10, 5, 5))
opt = torch.optim.Adam(model.parameters())

def forward(model, nodeset, train=True):
    if train:
        return model(nodeset)
    else:
        with torch.no_grad():
            return model(nodeset)

def run(edge_set, train=True):
    if train:
        model.train()
    else:
        model.eval()

    edge_batches = edge_set[torch.randperm(edge_set.shape[0])].split(batch_size)

    with tqdm.tqdm(edge_batches) as tq:
        sum_loss = 0
        for batch_id, batch in enumerate(tq):
            src, dst = g.find_edges(batch)
            dst_neg = torch.LongTensor(
                    [np.random.choice(neighbors[i.item()]) for i in dst])

            # filter the nodes to those showing up in g_prior
            src_np = src.numpy()
            dst_np = dst.numpy()
            dst_neg_np = dst_neg.numpy()
            np_mask = (np.isin(src_np, g_prior_nid_np) &
                       np.isin(dst_np, g_prior_nid_np) &
                       np.isin(dst_neg, g_prior_nid_np))
            src = torch.from_numpy(src_np[np_mask])
            dst = torch.from_numpy(dst_np[np_mask])
            dst_neg = torch.from_numpy(dst_neg_np[np_mask])

            src_prior = g_prior.map_to_subgraph_nid(src)
            dst_prior = g_prior.map_to_subgraph_nid(dst)
            dst_neg_prior = g_prior.map_to_subgraph_nid(dst_neg)

            nodeset = cuda(torch.cat([src_prior, dst_prior, dst_neg_prior]))
            src_size, dst_size, dst_neg_size = \
                    src.shape[0], dst.shape[0], dst_neg.shape[0]
            h_src, h_dst, h_dst_neg = (
                    forward(model, nodeset, train)
                    .split([src_size, dst_size, dst_neg_size]))

            loss = ((h_src * (h_dst_neg - h_dst)).sum(1) + 1).clamp(min=0).mean()
            assert loss.item() == loss.item()

            if train:
                model.zero_grad()
                loss.backward()
                for name, p in model.named_parameters():
                    assert (p.grad != p.grad).sum() == 0
                opt.step()

            sum_loss += loss.item()
            tq.set_postfix({'loss': '%.6f' % loss.item(),
                            'avg_loss': sum_loss / (batch_id + 1)})

for epoch in range(500):
    run(g_train_edges, True)
    run(g_valid_edges, False)
