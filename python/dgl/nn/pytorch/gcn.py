"""
Semi-Supervised Classification with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1609.02907
Code: https://github.com/tkipf/gcn

GCN with SPMV specialization.
"""
import torch.nn as nn

import dgl
import dgl.function as fn
from dgl.base import ALL, is_all

class NodeUpdateModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super(NodeUpdateModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.attribute = None

    def forward(self, node):
        h = self.linear(node['accum'])
        if self.activation:
            h = self.activation(h)
        if self.attribute:
            return {self.attribute: h}
        else:
            return h

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout=0):
        super(GCN, self).__init__()
        self.dropout = dropout
        # input layer
        self.update_func = NodeUpdateModule(in_feats, out_feats, activation)

    def forward(self, g, u=ALL, v=ALL, attribute=None):
        if is_all(u) and is_all(v):
            g.update_all(fn.copy_src(src=attribute),
                         fn.sum(out='accum'),
                         self.update_func,
                         batchable=True)
        else:
            g.send_and_recv(u, v,
                            fn.copy_src(src=attribute),
                            fn.sum(out='accum'),
                            self.update_func,
                            batchable=True)
        g.pop_n_repr('accum')
        return g
