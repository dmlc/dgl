#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import dgl.function as fn
from dgl.nn.pytorch import GATConv

class GraphConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, bias=True):
        super(GraphConvLayer, self).__init__()
        self.mlp = nn.Linear(in_feats * 2, out_feats, bias=bias)

    def forward(self, bipartite, feat):
        if isinstance(feat, tuple):
            srcfeat, dstfeat = feat
        else:
            srcfeat = feat
            dstfeat = feat[:graph.num_dst_nodes()]
        graph = bipartite.local_var()

        graph.srcdata['h'] = srcfeat
        graph.update_all(fn.u_mul_e('h', 'affine', 'm'),
                         fn.sum(msg='m', out='h'))

        gcn_feat = torch.cat([dstfeat, graph.dstdata['h']], dim=-1)
        out = self.mlp(gcn_feat)
        return out

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0, use_GAT = False, K = 1):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if use_GAT:
            self.gcn_layer = GATConv(in_dim, out_dim, K, allow_zero_in_degree = True)
            self.bias = nn.Parameter(torch.Tensor(K, out_dim))
            init.constant_(self.bias, 0)
        else:
            self.gcn_layer = GraphConvLayer(in_dim, out_dim, bias=True)

        self.dropout = dropout
        self.use_GAT = use_GAT

    def forward(self, bipartite, features):
        out = self.gcn_layer(bipartite, features)

        if self.use_GAT:
            out = torch.mean(out + self.bias, dim = 1)

        out = out.reshape(out.shape[0], -1)
        out = F.relu(out)
        if self.dropout > 0:
            out = F.dropout(out, self.dropout, training=self.training)

        return out
