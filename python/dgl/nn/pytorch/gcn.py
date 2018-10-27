"""
Semi-Supervised Classification with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1609.02907
Code: https://github.com/tkipf/gcn

GCN with SPMV specialization.
"""
import torch.nn as nn

from ... import function as fn
from ...base import ALL, is_all


class NodeUpdateModule(nn.Module):
    def __init__(self, node_field, in_feats, out_feats, activation=None,
                 dropout=0):
        super(NodeUpdateModule, self).__init__()

        self.node_field = node_field

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.

        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        if self.dropout:
            h = self.dropout(node[self.node_field])
        else:
            h = node[self.node_field]

        h = self.linear(h)
        if self.activation:
            h = self.activation(h)

        return {self.node_field: h}

class GraphConvolutionLayer(nn.Module):
    """Single graph convolution layer as in https://arxiv.org/abs/1609.02907."""
    def __init__(self,
                 g,
                 node_field,
                 in_feats,
                 out_feats,
                 activation,
                 dropout=0):
        """
        node_filed: hashable keys for node features, e.g. 'h'
        msg_field: hashable keys for message features, e.g. 'm'. In GCN, this is
            just AH, where A is the adjacency matrix and H is current node features.
        """
        super(GraphConvolutionLayer, self).__init__()
        self.g = g

        self.node_field = node_field

        # input layer
        self.update_func = NodeUpdateModule(node_field, in_feats, out_feats,
                                            activation, dropout)

    def forward(self, features, u=ALL, v=ALL):
        self.g.set_n_repr({self.node_field : features})

        if is_all(u) and is_all(v):
            self.g.update_all(fn.copy_src(src=self.node_field, out='m'),
                              fn.sum(msg='m', out=self.node_field),
                              self.update_func)
        else:
            self.g.send_and_recv(u, v,
                                 fn.copy_src(src=self.node_field, out='m'),
                                 fn.sum(msg='m', out=self.node_field),
                                 self.update_func)
        return self.g.pop_n_repr(self.node_field)
