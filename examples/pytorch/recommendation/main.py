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
g_test_edges = g.filter_edges(lambda edges: edges.data['test'])
g_prior = g.edge_subgraph(g_prior_edges)
g_train = g.edge_subgraph(g_train_edges)
g_prior_nid = g_prior.parent_nid

model = cuda(PinSage(g_prior, [n_hidden] * n_layers, 10, 5, 5))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


def forward(model, nodeset, train=True):
    if train:
        return model(nodeset)
    else:
        with torch.no_grad():
            return model(nodeset)


def filter_nid(nids, nid_from):
    nids = [nid.numpy() for nid in nids]
    nid_from = nid_from.numpy()
    np_mask = np.logical_and(*[np.isin(nid, nid_from) for nid in nids])
    return [torch.from_numpy(nid[np_mask]) for nid in nids]


def train(edge_set):
    model.train()

    edge_batches = edge_set[torch.randperm(edge_set.shape[0])].split(batch_size)

    with tqdm.tqdm(edge_batches) as tq:
        sum_loss = 0
        sum_acc = 0
        count = 0
        for batch_id, batch in enumerate(tq):
            count += batch.shape[0]
            src, dst = g.find_edges(batch)
            dst_neg = torch.LongTensor(
                    [np.random.choice(neighbors[i.item()]) for i in dst])

            src, dst, dst_neg = filter_nid([src, dst, dst_neg], g_prior_nid)

            src_prior = g_prior.map_to_subgraph_nid(src)
            dst_prior = g_prior.map_to_subgraph_nid(dst)
            dst_neg_prior = g_prior.map_to_subgraph_nid(dst_neg)

            nodeset = cuda(torch.cat([src_prior, dst_prior, dst_neg_prior]))
            src_size, dst_size, dst_neg_size = \
                    src.shape[0], dst.shape[0], dst_neg.shape[0]

            h_src, h_dst, h_dst_neg = (
                    forward(model, nodeset, True)
                    .split([src_size, dst_size, dst_neg_size]))

            diff = (h_src * (h_dst_neg - h_dst)).sum(1)
            loss = (diff + 1).clamp(min=0).mean()
            acc = (diff < 0).sum()
            assert loss.item() == loss.item()

            opt.zero_grad()
            loss.backward()
            for name, p in model.named_parameters():
                assert (p.grad != p.grad).sum() == 0
            opt.step()

            sum_loss += loss.item()
            sum_acc += acc.item()
            avg_loss = sum_loss / (batch_id + 1)
            avg_acc = sum_acc / count
            tq.set_postfix({'loss': '%.6f' % loss.item(),
                            'avg_loss': avg_loss,
                            'avg_acc': avg_acc})

    return avg_loss, avg_acc


def runtest(edgeset):
    model.eval()

    n_users = len(ml.users.index)
    n_items = len(ml.movies.index)

    with torch.no_grad():
        for uid in range(n_users):
            dst = torch.arange(n_users, n_users + n_items)
            src = torch.zeros_like(dst).fill_(uid)
            src, dst = filter_nid([src, dst], g_prior_nid)

            src_prior = g_prior.map_to_subgraph_nid(src)
            dst_prior = g_prior.map_to_subgraph_nid(dst)

            nodeset = cuda(torch.cat([src_prior, dst_prior]))
            src_size, dst_size = src.shape[0], dst.shape[0]

            h_src, h_dst = (
                    forward(model, nodeset, True)
                    .split([src_size, dst_size]))

            score = (h_src * h_dst).sum(1)


def train():
    best_acc = 0
    for epoch in range(500):
        print('Epoch %d train' % epoch)
        run(g_train_edges, True)
        print('Epoch %d validation' % epoch)
        with torch.no_grad():
            valid_loss, valid_acc = run(g_valid_edges, False)
            if best_acc < valid_acc:
                best_acc = valid_acc
                torch.save(model.state_dict(), 'model.pt')
        print('Epoch %d test' % epoch)
        with torch.no_grad():
            test_loss, test_acc = run(g_test_edges, False)


if __name__ == '__main__':
    train()
