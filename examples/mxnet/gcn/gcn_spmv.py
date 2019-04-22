"""GCN using builtin functions that enables SPMV optimization.

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import math
import mxnet as mx
from mxnet import gluon
import dgl
import dgl.function as fn

class GCNLayer(gluon.Block):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.g = g
        with self.name_scope():
            stdv = 1. / math.sqrt(out_feats)
            self.weight = self.params.get('weight', shape=(in_feats, out_feats),
                    init=mx.init.Uniform(stdv))
            if bias:
                self.bias = self.params.get('bias', shape=(out_feats,),
                    init=mx.init.Uniform(stdv))
            else:
                self.bias = None
        self.activation = activation
        self.dropout = dropout

    def forward(self, h):
        if self.dropout:
            h = mx.nd.Dropout(h, p=self.dropout)
        h = mx.nd.dot(h, self.weight.data(h.context))
        # normalization by square root of src degree
        h = h * self.g.ndata['norm']
        self.g.ndata['h'] = h
        self.g.update_all(fn.copy_src(src='h', out='m'),
                          fn.sum(msg='m', out='h'))
        h = self.g.ndata.pop('h')
        # normalization by square root of dst degree
        h = h * self.g.ndata['norm']
        # bias
        if self.bias is not None:
            h = h + self.bias.data(h.context)
        if self.activation:
            h = self.activation(h)
        return h

class GCN(gluon.Block):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = gluon.nn.Sequential()
        # input layer
        self.layers.add(GCNLayer(g, in_feats, n_hidden, activation, 0.))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.add(GCNLayer(g, n_hidden, n_hidden, activation, dropout))
        # output layer
        self.layers.add(GCNLayer(g, n_hidden, n_classes, None, dropout))


    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h
