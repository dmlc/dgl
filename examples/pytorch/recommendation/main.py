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
neighbors = ml.user_neighbors + ml.movie_neighbors

n_hidden = 100
n_layers = args.layers
batch_size = 256
margin = 0.9

n_negs = args.n_negs
hard_neg_prob = args.hard_neg_prob

sched_lambda = {
        'none': lambda epoch: 1,
        'decay': lambda epoch: max(0.98 ** epoch, 1e-4),
        }
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
sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_lambda[args.sched])


def forward(model, g_prior, nodeset, train=True):
    if train:
        return model(g_prior, nodeset)
    else:
        with torch.no_grad():
            return model(g_prior, nodeset)


def filter_nid(nids, nid_from):
    nids = [nid.numpy() for nid in nids]
    nid_from = nid_from.numpy()
    np_mask = np.logical_and(*[np.isin(nid, nid_from) for nid in nids])
    return [torch.from_numpy(nid[np_mask]) for nid in nids]


def runtrain(g_prior_edges, g_train_edges, train):
    global opt
    if train:
        model.train()
    else:
        model.eval()

    g_prior_src, g_prior_dst = g.find_edges(g_prior_edges)
    g_prior = DGLGraph()
    g_prior.add_nodes(g.number_of_nodes())
    g_prior.add_edges(g_prior_src, g_prior_dst)
    g_prior.ndata.update({k: cuda(v) for k, v in g.ndata.items()})
    edge_batches = g_train_edges[torch.randperm(g_train_edges.shape[0])].split(batch_size)

    with tqdm.tqdm(edge_batches) as tq:
        sum_loss = 0
        sum_acc = 0
        count = 0
        for batch_id, batch in enumerate(tq):
            count += batch.shape[0]
            src, dst = g.find_edges(batch)
            dst_neg = []
            for i in range(len(dst)):
                if np.random.rand() < hard_neg_prob:
                    nb = torch.LongTensor(neighbors[dst[i].item()])
                    mask = ~(g.has_edges_between(nb, src[i].item()).byte())
                    dst_neg.append(np.random.choice(nb[mask].numpy(), n_negs))
                else:
                    dst_neg.append(np.random.randint(
                        len(ml.user_ids), len(ml.user_ids) + len(ml.movie_ids), n_negs))
            dst_neg = torch.LongTensor(dst_neg)
            dst = dst.view(-1, 1).expand_as(dst_neg).flatten()
            src = src.view(-1, 1).expand_as(dst_neg).flatten()
            dst_neg = dst_neg.flatten()

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


def runtest(g_prior_edges, validation=True):
    model.eval()

    n_users = len(ml.users.index)
    n_items = len(ml.movies.index)

    g_prior_src, g_prior_dst = g.find_edges(g_prior_edges)
    g_prior = DGLGraph()
    g_prior.add_nodes(g.number_of_nodes())
    g_prior.add_edges(g_prior_src, g_prior_dst)
    g_prior.ndata.update({k: cuda(v) for k, v in g.ndata.items()})

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
        g_prior_edges = g.filter_edges(lambda edges: edges.data['prior'])
        g_train_edges = g.filter_edges(lambda edges: edges.data['train'] & ~edges.data['inv'])
        g_prior_train_edges = g.filter_edges(
                lambda edges: edges.data['prior'] | edges.data['train'])

        print('Epoch %d validation' % epoch)
        with torch.no_grad():
            valid_mrr = runtest(g_prior_train_edges, True)
            if best_mrr < valid_mrr.mean():
                best_mrr = valid_mrr.mean()
                torch.save(model.state_dict(), 'model.pt')
        print(pd.Series(valid_mrr).describe())
        print('Epoch %d test' % epoch)
        with torch.no_grad():
            test_mrr = runtest(g_prior_train_edges, False)
        print(pd.Series(test_mrr).describe())

        print('Epoch %d train' % epoch)
        runtrain(g_prior_edges, g_train_edges, True)

        if epoch == args.sgd_switch:
            opt = torch.optim.SGD(model.parameters(), lr=0.6)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_lambda['decay'])
        elif epoch < args.sgd_switch:
            sched.step()


if __name__ == '__main__':
    train()
