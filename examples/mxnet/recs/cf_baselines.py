import mxnet as mx
import numpy as np
from dgl import DGLGraph
from mxnet import gluon
import functools

# import argparse
# parser = argparse.ArgumentParser(description='GCMC')
# parser.add_argument("--dropout", type=float, default=0.2,
#         help="dropout probability")
# parser.add_argument("--gpu", type=int, default=-1,
#         help="gpu")
# parser.add_argument("--lr", type=float, default=0.01,
#         help="learning rate")
# parser.add_argument("--wd", type=float, default=1e-4,
#         help="weight decay")
# parser.add_argument("--n-epochs", type=int, default=200,
#         help="number of training epochs")
# parser.add_argument("--n-hidden", type=int, default=6,
#         help="number of hidden gcn units")
# parser.add_argument("--n-layers", type=int, default=1,
#         help="number of hidden gcn layers")
# parser.add_argument("--normalization",
#         choices=['sym','left'], default=None,
#         help="graph normalization types (default=None)")


# G.update_all(
#     lambda edge : {
#         # n_edges, n_classes
#         'm': edge.data['r'] * edge.data['is_train'].expand_dims(1)
#     },
#     lambda node : {
#         'degree' : mx.nd.sum(node.mailbox['m'], axis=1)
#         # n_nodes, n_classes
#     },
# )

# def linear_naive(edge):
#     src = edge.src['h'] / (edge.src['h'].sum(axis=1, keepdims=True) + 1e-10)
#     dst = edge.dst['h'] / (edge.dst['h'].sum(axis=1, keepdims=True) + 1e-10)
#     return {'pred' : (src+dst)/2.}


def gcn_msg(edge, share_weights=False):
    if share_weights:
        W = mx.nd.SequenceMask(edge.src['W'], # class, hidden
            sequence_length = edge.data['r']+1, # class
            use_sequence_length=True, axis=1
        ).sum(axis=1)
    else:
        W = mx.nd.SequenceLast(edge.src['W'], # class, hidden
            sequence_length = edge.data['r']+1, # class
            use_sequence_length=True, axis=1
        )
    return {
        'W'   : mx.nd.where(edge.data['is_train'], W, mx.nd.zeros_like(W)),
        'is_train' : edge.data['is_train'],
    }


def gcn_reduce(node):
    accum = mx.nd.sum(node.mailbox['W'], axis=1) # hidden
    degree = mx.nd.sum(node.mailbox['is_train'], axis=1)
    accum = accum / (degree + 1e-10).expand_dims(1)
    return {'accum' : accum}


class NodeUpdateModule(gluon.Block):
    def __init__(self, out_feats, activation=None, dropout=0):
        super(NodeUpdateModule, self).__init__()
        self.linear = gluon.nn.Dense(out_feats, activation=activation)
        self.dropout = gluon.nn.Dropout(dropout)


    def forward(self, node):
        accum = self.linear(node.data['accum'])              # hidden
        h = self.linear(node.data['c'] + node.data['accum']) # hidden
        h = self.dropout(h)
        return {'h': h}


def _cumsum(arr, psum):
    return arr+psum, arr+psum


def quad_fcn(edge, share_weights=False):
    F = mx.nd
    user2item = edge.data['user2item']
    item_b = F.where(user2item, edge.dst['b'], edge.src['b']) # class
    user_h = F.where(user2item, edge.src['h'], edge.dst['h']) # hidden
    item_V = F.where(user2item, edge.dst['W'], edge.src['W']) # class, hidden

    score = item_b + F.batch_dot(item_V,
                        user_h.expand_dims(2)).squeeze(axis=2) # class

    if share_weights:
        _score = score.T
        _score, _ = F.contrib.foreach(_cumsum, _score, F.zeros_like(_score[0]))
        score = _score.T

    return {'score' : score}


def ordinal_transform(score):
    _n = score.shape[1]

    triu = mx.nd.array(np.triu(np.ones((_n, _n))), ctx=score.context)
    tril = mx.nd.array(np.tril(np.ones((_n, _n))), ctx=score.context)

    log_softmax_left = mx.nd.concat(*[
        score.slice_axis(axis=1, begin=0, end=j+1).log_softmax().slice_axis(axis=1, begin=-1, end=None)
        for j in range(_n)
    ], dim=1)

    log_softmax_right = mx.nd.concat(*[
        score.slice_axis(axis=1, begin=j, end=None).log_softmax().slice_axis(axis=1, begin=0, end=1)
        for j in range(_n)
    ], dim=1)

    return mx.nd.dot(log_softmax_left, triu) + mx.nd.dot(log_softmax_right, tril)


class CF_NADE(gluon.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 share_weights=False,
                 ):
        super(CF_NADE, self).__init__()
        self.n_hidden   = n_hidden
        self.n_classes  = n_classes
        self.dropout    = dropout

        with self.name_scope():
            self.W = self.params.get('W', shape=(in_feats, n_classes, n_hidden))
            self.b = self.params.get('b', shape=(in_feats, n_classes))
            self.c = self.params.get('c', shape=(in_feats, n_hidden))
            self.dropout = gluon.nn.Dropout(dropout)

            self.gcn_msg = functools.partial(gcn_msg,
                                    share_weights=share_weights)
            self.conv_layers = gluon.nn.Sequential()
            for i in range(n_layers):
                self.conv_layers.add(
                    NodeUpdateModule(n_hidden, activation, dropout))
            self.quad_fcn = functools.partial(quad_fcn,
                                    share_weights=share_weights)


    def forward(self, g):
        g.ndata['W'] = self.dropout(self.W.data())
        g.ndata['b'] = self.b.data() # movie bias
        g.ndata['c'] = self.c.data() # global bias

        # g.ndata['h'] = (g.ndata['degree'] /
        #     (g.ndata['degree'].sum(axis=1, keepdims=True) + 1e-10)
        # ).expand_dims(2)

        for layer in self.conv_layers:
            g.update_all(self.gcn_msg, gcn_reduce, layer)

        g.apply_edges(self.quad_fcn)

        return g.edata.pop('score')

