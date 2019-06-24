import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tqdm
from rec.model.graphsage import GraphSage
from rec.model.layers import ScaledEmbedding, ZeroEmbedding
from rec.utils import cuda
from rec.comm.receiver import NodeFlowReceiver
from dgl import DGLGraph
from validation import *
import operator

import argparse
import pickle
import os
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, default='SGD')
parser.add_argument('--lr', type=float, default=1)
parser.add_argument('--n', type=float, default=500)
parser.add_argument('--sched', type=str, default='none',
                    help='learning rate scheduler (none or decay)')
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--use-feature', action='store_true')
parser.add_argument('--sgd-switch', type=int, default=-1,
                    help='The number of epoch to switch to SGD (-1 = never)')
parser.add_argument('--n-negs', type=int, default=1)
parser.add_argument('--cache', type=str, default='/tmp/dataset.pkl',
                    help='File to cache the postprocessed dataset object')
parser.add_argument('--dataset', type=str, default='movielens')
parser.add_argument('--raw-dataset-path', type=str, default='/efs/quagan/movielens')
parser.add_argument('--train-port', type=int, default=5902)
parser.add_argument('--valid-port', type=int, default=5901)
parser.add_argument('--num-samplers', type=int, default=8)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--l2', type=float, default=1e-9)
parser.add_argument('--loss', type=str, default='bpr')
args = parser.parse_args()

print(args)

ml = load_data(args)

# Generate a mask for
# Prior graph - where the interactions are known and based upon during training
# Training set - the interactions where minibatches are sampled and losses are computed
# Validation set
# Test set
# Currently, the prior graph and the training set are the same.  This makes the training
# prone to overfitting and will be replaced with edge masking later.

# The dictionary that maps the dataset to evaluation functions
_compute_validation = {
        'movielens1m': compute_validation_rating,
        'movielens10m': compute_validation_rating,
        'movielens10m-imp': compute_validation_imp,
        'movielens1m-imp': compute_validation_imp,
        }
# The function that aggregates per-example metric into one single number
_metric_agg = {
        'movielens1m': lambda x: np.sqrt(np.mean(x)),
        'movielens10m': lambda x: np.sqrt(np.mean(x)),
        'movielens10m-imp': np.mean,
        'movielens1m-imp': np.mean,
        }
# The function that determines whether the first metric is better than the
# second metric
_better = {
        'movielens1m': operator.lt,
        'movielens10m': operator.lt,
        'movielens10m-imp': operator.gt,
        'movielens1m-imp': operator.gt,
        }
# The evaluation function for the current dataset
compute_validation = _compute_validation[args.dataset]
metric_agg = _metric_agg[args.dataset]
better = _better[args.dataset]

g = ml.g
g_train_edges = g.filter_edges(lambda edges: edges.data['train'])
g_train = g.edge_subgraph(g_train_edges, True)
g_train.copy_from_parent()

n_hidden = 100
n_layers = args.layers
batch_size = args.batch_size
margin = 0.9

n_negs = args.n_negs

# Learning rate scheduler w.r.t. current epoch
sched_lambda = {
        'none': lambda epoch: 1,
        'decay': lambda epoch: max(0.98 ** epoch, 1e-4),
        }
# Loss function for implicit feedback
loss_func = {
        'hinge': lambda diff: (diff + margin).clamp(min=0).mean(),
        'bpr': lambda diff: (1 - torch.sigmoid(-diff)).mean(),
        }

emb = {}

# Arguably, the only difference between PinSage and GraphSage is the neighborhood
# of a node.
# PinSage constructs neighborhood from random walks, while GraphSage takes direct
# neighbors of a node.
model = cuda(GraphSage(
    [n_hidden] * (n_layers + 1),
    use_feature=args.use_feature,
    G=g_train,
    emb=emb,
    ))
# The learnable embeddings for each user and product that don't take part in GraphSage
# convolution.
# For now, only the portion of user embeddings (i.e. nid_h[1:len(ml.users)+1]) are
# effective.
# The product portion is reserved in case that one wishes to make GraphSage output a
# "residual" portion (i.e. instead of direct matrix factorization, one adds the GraphSage
# output to the user/product embedding nid_h before the dot product).
nid_h = cuda(ScaledEmbedding(1 + len(ml.users) + len(ml.products), n_hidden, padding_idx=0))
# The learnable embeddings for each user and product that take part in GraphSage
# convolution.
# For now, only the portion of product embeddings (i.e.
# nid_m[1+len(ml.users):1+len(ml.users)+len(ml.products)]) are effective.
nid_m = cuda(ScaledEmbedding(1 + len(ml.users) + len(ml.products), n_hidden, padding_idx=0))
# User/Product-specific biases.
nid_b = cuda(ZeroEmbedding(1 + len(ml.users) + len(ml.products), 1, padding_idx=0))

# Optimizer & learning rate scheduler
opt = getattr(torch.optim, args.opt)(
        list(model.parameters()) +
        list(nid_h.parameters()) +
        list(nid_b.parameters()) +
        list(nid_m.parameters()),
        lr=args.lr,
        weight_decay=args.l2)
sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_lambda[args.sched])

def forward(model, nodeflow, train=True):
    if train:
        return model(nodeflow, nid_m)
    else:
        with torch.no_grad():
            return model(nodeflow, nid_m)

# I wrote my own NodeFlow receiver and sender instead of the DGL builtin ones, since
# I need to send additional data along with the NodeFlow.
train_sampler = NodeFlowReceiver(args.train_port)
train_sampler.waitfor(args.num_samplers)
train_sampler.set_parent_graph(g_train)
valid_sampler = NodeFlowReceiver(args.valid_port)
valid_sampler.waitfor(args.num_samplers)
valid_sampler.set_parent_graph(g_train)

def runtrain():
    global opt
    model.train()

    # The edges we are going to generate training examples from are always
    # from user to product, without the inverse direction.
    # Essentially, only the product embeddings are passed through GraphSage;
    # user embeddings would be left untouched.
    # We distribute the edges to train evenly to the remote samplers.
    edge_shuffled = torch.LongTensor(
            np.random.permutation(
                g_train.filter_edges(
                    lambda edges: ~edges.data['inv']).numpy()))
    train_sampler.distribute(edge_shuffled.numpy())

    train_sampler_iter = iter(train_sampler)

    with tqdm.tqdm(train_sampler_iter) as tq:
        sum_loss = 0
        sum_acc = 0
        count = 0
        for batch_id, (nodeflow, aux_data) in enumerate(tq):
            edges, src, dst, dst_neg = aux_data[:4]
            edges = torch.LongTensor(edges)
            src = torch.LongTensor(src)
            dst = torch.LongTensor(dst)
            src_size = dst_size = src.shape[0]
            count += src_size

            if dst_neg is not None:
                dst_neg = torch.LongTensor(dst_neg)
                dst_neg_size = dst_neg.shape[0]
                nodeset = torch.cat([dst, dst_neg])
            else:
                nodeset = dst

            nodeflow.copy_from_parent(edge_embed_names=None)

            # The features on nodeflow is stored on CPUs for now.  We copy them to GPUs
            # in model.forward().
            node_output = forward(model, nodeflow, True)

            # Reindex the nodeflow output to figure out which outputs are from users (not used),
            # products, and "negative" products.
            output_idx = nodeflow.map_from_parent_nid(-1, nodeset, True)
            h = node_output[output_idx]
            if dst_neg is not None:
                h_dst, h_dst_neg = h.split([dst_size, dst_neg_size])
            else:
                h_dst = h
            h_src = nid_h(cuda(src + 1))

            b_src = nid_b(cuda(src + 1)).squeeze()
            b_dst = nid_b(cuda(dst + 1)).squeeze()
            if dst_neg is not None:
                b_dst_neg = nid_b(cuda(dst_neg + 1)).view(src_size, n_negs).squeeze()

            # Compute scores, losses, and classification accuracies
            pos_score = (h_src * h_dst).sum(1) + b_src + b_dst
            pos_nlogp = -F.logsigmoid(pos_score)
            if dst_neg is not None:
                neg_score = (h_src[:, None] * h_dst_neg.view(src_size, n_negs, -1)).sum(2)
                neg_score = neg_score + b_src[:, None] + b_dst_neg
                neg_nlogp = -F.logsigmoid(-neg_score)
            if args.dataset.startswith('movielens') and not args.dataset.endswith('imp'):
                # rating prediction - L2 loss
                loss = (pos_score - cuda(g_train.edges[edges].data['rating'])) ** 2
                loss = loss.mean()
                acc = loss
            else:
                # link prediction - BPR or hinge loss
                diff = neg_score.mean(1) - pos_score
                loss = loss_func[args.loss](diff)
                # Or use the following for NCE loss
                #loss = (pos_nlogp + neg_nlogp.sum(1)).mean()
                acc = ((pos_score > 0).sum() + (neg_score < 0).sum()).float() / (src_size * (1 + n_negs))

            opt.zero_grad()
            loss.backward()
            opt.step()

            sum_loss += loss.item()
            sum_acc += acc.item()
            avg_loss = sum_loss / (batch_id + 1)
            avg_acc = sum_acc / (batch_id + 1)
            tq.set_postfix({'loss': '%.6f' % loss.item(),
                            'avg_loss': '%.3f' % avg_loss,
                            'avg_acc': '%.3f' % avg_acc})
            tq.update()

    return avg_loss, avg_acc

def runtest(validation=True):
    model.eval()

    n_users = len(ml.user_ids)
    n_items = len(ml.product_ids)

    valid_sampler.distribute(np.arange(n_items))
    valid_sampler_iter = iter(valid_sampler)

    hs = []
    auxs = []
    with torch.no_grad():
        with tqdm.tqdm(valid_sampler_iter) as tq:
            for nodeflow, aux_data in tq:
                nodeflow.copy_from_parent(edge_embed_names=None)
                h = forward(model, nodeflow, False)
                hs.append(h)
                auxs.append(torch.LongTensor(aux_data))
    h = torch.cat(hs, 0)
    auxs = torch.cat(auxs, 0)
    assert (np.sort(auxs.numpy()) == np.arange(n_items)).all()
    h = h[auxs.sort()[1]]     # reorder h

    h = torch.cat([
        nid_h(cuda(torch.arange(0, n_users).long() + 1)),
        h], 0)
    b = nid_b(cuda(torch.arange(1, 1 + n_users + n_items)))

    return compute_validation(ml, h, b, model, not validation)


def train():
    global opt, sched
    best_metric = None
    best_test = 0

    for epoch in range(args.n):
        print('Epoch %d validation' % epoch)

        with torch.no_grad():
            valid_metric = runtest(True)
        print(pd.Series(valid_metric).describe())
        print('Epoch %d test' % epoch)
        with torch.no_grad():
            test_metric = runtest(False)
        if best_metric is None or not better(best_metric, metric_agg(valid_metric)):
            best_metric = metric_agg(valid_metric)
            best_test = metric_agg(test_metric)
            torch.save(model.state_dict(), 'model.pt')
        print(pd.Series(test_metric).describe())
        print('Best valid:', best_metric, best_test)

        print('Epoch %d train' % epoch)
        runtrain()

        if epoch == args.sgd_switch:
            opt = torch.optim.SGD(model.parameters(), lr=0.6)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_lambda['decay'])
        elif epoch < args.sgd_switch:
            sched.step()

    print('Best valid:', best_metric, best_test)


if __name__ == '__main__':
    train()
