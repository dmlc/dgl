#-*- coding:utf-8 -*-

import torch as th
import torch.nn as nn
import torch.functional as F

import dgl
import dgl.function as fn


class dummy_layer(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(dummy_layer, self).__init__()
        self.layer = nn.linear(in_dim * 2, out_dim, bias=True)

    def forward(self, graph, n_feats, e_weights):
        graph.edata['ew'] = e_weights
        graph.ndata['h'] = n_feats
        graph.update_all(fn.u_mul_e('h', 'ew', 'm'), fn.mean('m', 'h'))

        graph.ndata['h'] = self.layer(th.cat([graph.ndata['h'], n_feats], dim=-1))

        output = F.relu(graph.ndata['h'])
        return output


class dummy_gnn_model(nn.Module):

    """
    A dummy gnn model, which is same as graph sage, but could adopt edge mask in forward

    """
    edge_weight_cont = 1.0

    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim):
        super(dummy_gnn_model, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.in_layer = dummy_layer(self.in_dim, self.hid_dim)
        self.hid_layer = dummy_layer(self.hid_dim, self.hid_dim)
        self.out_layer = dummy_layer(self.hid_dim, self.out_dim)

    def forward(self, graph, n_feat, edge_weights):

        h = self.in_layer(graph, n_feat, edge_weights)
        h = self.hid_layer(graph, h, edge_weights)
        h = self.out_layer(graph, h, edge_weights)

        return h