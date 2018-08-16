"""
Semi-Supervised Classification with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1609.02907
Code: https://github.com/tkipf/gcn

GCN with SPMV specialization.
"""
import torch.nn as nn
from dgl.base import ALL, is_all

class NodeUpdateModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super(NodeUpdateModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.attribute = None

    def set_attribute_to_update(self, attribute):
        self.attribute = attribute

    def forward(self, node, accum, attribute=None):
        if self.attribute:
            accum = accum[self.attribute]
        h = self.linear(accum)
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
        self.update_func.set_attribute_to_update(attribute)
        if is_all(u) and is_all(v):
            g.update_all('from_src', 'sum', self.update_func, batchable=True)
        else:
            g.update_by_edge(u, v, 'from_src', 'sum', self.update_func, batchable=True)
        return g
