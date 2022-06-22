"""GCN using basic message passing

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import argparse, time, math
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset


def gcn_msg(edge):
    msg = edge.src['h'] * edge.src['norm']
    return {'m': msg}


def gcn_reduce(node):
    accum = torch.sum(node.mailbox['m'], 1) * node.data['norm']
    return {'h': accum}


class NodeApplyModule(nn.Module):
    def __init__(self, out_feats, activation=None, bias=True):
        super(NodeApplyModule, self).__init__()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, nodes):
        h = nodes.data['h']
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return {'h': h}


class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.node_update = NodeApplyModule(out_feats, activation, bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        g.ndata['h'] = torch.mm(h, self.weight)
        g.update_all(gcn_msg, gcn_reduce, self.node_update)
        h = g.ndata.pop('h')
        return h
