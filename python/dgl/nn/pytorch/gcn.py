"""
Semi-Supervised Classification with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1609.02907
Code: https://github.com/tkipf/gcn

GCN with SPMV specialization.
"""
import torch.nn as nn

class NodeUpdateModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super(NodeUpdateModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node, accum):
        h = self.linear(accum)
        if self.activation:
            h = self.activation(h)
        return h

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        # input layer
        self.update_func = NodeUpdateModule(in_feats, out_feats, activation)

    def forward(self, g):
        g.update_all('from_src', 'sum', self.update_func, batchable=True)
        return g
