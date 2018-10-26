"""
Semi-Supervised Classification with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1609.02907
Code: https://github.com/tkipf/gcn

GCN with SPMV specialization.
"""
import torch.nn as nn
import torch.nn.functional as F

from ... import function as fn
from ...base import ALL, is_all


class NodeUpdateModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super(NodeUpdateModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node['h'])
        if self.activation:
            h = self.activation(h)
        return {'h' : h}

class GraphConvolutionLayer(nn.Module):
    """Single graph convolution layer as in https://arxiv.org/abs/1609.02907.
    Adjacency matrix normalization not supported yet."""
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout=0):
        super(GraphConvolutionLayer, self).__init__()
        self.g = g
        self.dropout = dropout
        # input layer
        self.update_func = NodeUpdateModule(in_feats, out_feats, activation)

    def forward(self, features, u=ALL, v=ALL, attribute=None):
        self.g.set_n_repr({'h' : features})
        if self.dropout:
            if is_all(u):
                self.g.apply_nodes(apply_node_func=lambda node: {'h': F.dropout(node['h'], p=self.dropout,
                                                                                training=self.training)})
            else:
                self.g.apply_nodes(u, apply_node_func=lambda node: {'h': F.dropout(node['h'], p=self.dropout,
                                                                                   training=self.training)})

        if is_all(u) and is_all(v):
            self.g.update_all(fn.copy_src(src='h', out='m'),
                              fn.sum(msg='m', out='h'),
                              self.update_func)
        else:
            self.g.send_and_recv(u, v,
                                 fn.copy_src(src='h', out='m'),
                                 fn.sum(msg='m', out='h'),
                                 self.update_func)
        return self.g.pop_n_repr('h')
