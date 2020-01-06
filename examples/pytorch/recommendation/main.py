import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tqdm
from rec.model.pinsage import PinSage
from rec.datasets.movielens import MovieLens
from rec.utils import cuda
from dgl import DGLGraph

import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, default='SGD')
parser.add_argument('--lr', type=float, default=1)
parser.add_argument('--sched', type=str, default='none')
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--use-feature', action='store_true')
parser.add_argument('--sgd-switch', type=int, default=-1)
parser.add_argument('--n-negs', type=int, default=1)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--hard-neg-prob', type=float, default=0)
args = parser.parse_args()

print(args)

cache_file = 'ml.pkl'

if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        ml = pickle.load(f)
else:
    ml = MovieLens('./ml-1m')
    with open(cache_file, 'wb') as f:
        pickle.dump(ml, f)

g = ml.g

n_hidden = 100
n_layers = args.layers
batch_size = 256
margin = 0.9

n_negs = args.n_negs
hard_neg_prob = args.hard_neg_prob

loss_func = {
        'hinge': lambda diff: (diff + margin).clamp(min=0).mean(),
        'bpr': lambda diff: (1 - torch.sigmoid(-diff)).mean(),
        }

model = cuda(PinSage(
    g.number_of_nodes(),
    [n_hidden] * (n_layers + 1),
    20,
    0.5,
    10,
    use_feature=args.use_feature,
    G=g,
    ))
opt = getattr(torch.optim, args.opt)(model.parameters(), lr=args.lr)


def forward(model, g_prior, nodeset, train=True):
    if train:
        return model(g_prior, nodeset)
    else:
        with torch.no_grad():
            return model(g_prior, nodeset)


def runtrain(g_train_bases, g_train_pairs, train):
    global opt
    if train:
        model.train()
    else:
        model.eval()

    g_prior = g.edge_subgraph(g_train_bases, preserve_nodes=True)
    g_prior.copy_from_parent()

    # generate batches of training pairs
    edge_batches = g_train_pairs[torch.randperm(g_train_pairs.shape[0])].split(batch_size)

    with tqdm.tqdm(edge_batches) as tq:
        sum_loss = 0
        sum_acc = 0
        count = 0
        for batch_id, batch in enumerate(tq):
            count += batch.shape[0]
            # Get source (user) and destination (item) nodes, as well as negative items
            src, dst = g.find_edges(batch)
            dst_neg = []
            for i in range(len(dst)):
                dst_neg.append(np.random.randint(
                    len(ml.user_ids), len(ml.user_ids) + len(ml.movie_ids), n_negs))
            dst_neg = torch.LongTensor(dst_neg)
            dst = dst.view(-1, 1).expand_as(dst_neg).flatten()
            src = src.view(-1, 1).expand_as(dst_neg).flatten()
            dst_neg = dst_neg.flatten()

            # make sure that the source/destination/negative nodes have successors
            mask = (g_prior.in_degrees(dst_neg) > 0) & \
                   (g_prior.in_degrees(dst) > 0) & \
                   (g_prior.in_degrees(src) > 0)
            src = src[mask]
            dst = dst[mask]
            dst_neg = dst_neg[mask]
            if len(src) == 0:
                continue

            nodeset = cuda(torch.cat([src, dst, dst_neg]))
            src_size, dst_size, dst_neg_size = \
                    src.shape[0], dst.shape[0], dst_neg.shape[0]

            # get representations and compute losses
            h_src, h_dst, h_dst_neg = (
                    forward(model, g_prior, nodeset, train)
                    .split([src_size, dst_size, dst_neg_size]))

            diff = (h_src * (h_dst_neg - h_dst)).sum(1)
            loss = loss_func[args.loss](diff)
            acc = (diff < 0).sum()
            assert loss.item() == loss.item()

            grad_sqr_norm = 0
            if train:
                opt.zero_grad()
                loss.backward()
                for name, p in model.named_parameters():
                    assert (p.grad != p.grad).sum() == 0
                    grad_sqr_norm += p.grad.norm().item() ** 2
                opt.step()

            sum_loss += loss.item()
            sum_acc += acc.item() / n_negs
            avg_loss = sum_loss / (batch_id + 1)
            avg_acc = sum_acc / count
            tq.set_postfix({'loss': '%.6f' % loss.item(),
                            'avg_loss': '%.3f' % avg_loss,
                            'avg_acc': '%.3f' % avg_acc,
                            'grad_norm': '%.6f' % np.sqrt(grad_sqr_norm)})

    return avg_loss, avg_acc


def runtest(g_train_bases, ml, validation=True):
    model.eval()

    n_users = len(ml.users.index)
    n_items = len(ml.movies.index)

    g_prior = g.edge_subgraph(g_train_bases, preserve_nodes=True)
    g_prior.copy_from_parent()

    # Pre-compute the representations of users and items
    hs = []
    with torch.no_grad():
        with tqdm.trange(n_users + n_items) as tq:
            for node_id in tq:
                nodeset = cuda(torch.LongTensor([node_id]))
                h = forward(model, g_prior, nodeset, False)
                hs.append(h)
    h = torch.cat(hs, 0)

    rr = []

    with torch.no_grad():
        with tqdm.trange(n_users) as tq:
            for u_nid in tq:
                # For each user, exclude the items appearing in
                # (1) the training set, and
                # (2) either the validation set when testing, or the test set when
                #     validating.
                uid = ml.user_ids[u_nid]
                pids_exclude = ml.ratings[
                        (ml.ratings['user_id'] == uid) &
                        (ml.ratings['train'] | ml.ratings['test' if validation else 'valid'])
                        ]['movie_id'].values
                pids_candidate = ml.ratings[
                        (ml.ratings['user_id'] == uid) &
                        ml.ratings['valid' if validation else 'test']
                        ]['movie_id'].values
                pids = np.setdiff1d(ml.movie_ids, pids_exclude)
                p_nids = np.array([ml.movie_ids_invmap[pid] for pid in pids])
                p_nids_candidate = np.array([ml.movie_ids_invmap[pid] for pid in pids_candidate])

                # compute scores of items and rank them, then compute the MRR.
                dst = torch.from_numpy(p_nids) + n_users
                src = torch.zeros_like(dst).fill_(u_nid)
                h_dst = h[dst]
                h_src = h[src]

                score = (h_src * h_dst).sum(1)
                score_sort_idx = score.sort(descending=True)[1].cpu().numpy()

                rank_map = {v: i for i, v in enumerate(p_nids[score_sort_idx])}
                rank_candidates = np.array([rank_map[p_nid] for p_nid in p_nids_candidate])
                rank = 1 / (rank_candidates + 1)
                rr.append(rank.mean())
                tq.set_postfix({'rank': rank.mean()})

    return np.array(rr)


def train():
    global opt, sched
    best_mrr = 0
    for epoch in range(500):
        ml.refresh_mask()

        # In training, we perform message passing on edges marked with 'prior', and
        # do link prediction on edges marked with 'train'.
        # 'prior' and 'train' are disjoint so that the training pairs can not pass
        # messages between each other.
        # 'prior' and 'train' are re-generated everytime with ml.refresh_mask() above.
        g_train_bases = g.filter_edges(lambda edges: edges.data['prior'])
        g_train_pairs = g.filter_edges(lambda edges: edges.data['train'] & ~edges.data['inv'])
        # In testing we perform message passing on both 'prior' and 'train' edges.
        g_test_bases = g.filter_edges(
                lambda edges: edges.data['prior'] | edges.data['train'])

        print('Epoch %d validation' % epoch)
        with torch.no_grad():
            valid_mrr = runtest(g_test_bases, ml, True)
            if best_mrr < valid_mrr.mean():
                best_mrr = valid_mrr.mean()
                torch.save(model.state_dict(), 'model.pt')
        print(pd.Series(valid_mrr).describe())
        print('Epoch %d test' % epoch)
        with torch.no_grad():
            test_mrr = runtest(g_test_bases, ml, False)
        print(pd.Series(test_mrr).describe())

        print('Epoch %d train' % epoch)
        runtrain(g_train_bases, g_train_pairs, True)


if __name__ == '__main__':
    train()
