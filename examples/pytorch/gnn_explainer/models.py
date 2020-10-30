#-*- coding:utf-8 -*-

import torch as th
import torch.nn as nn

import dgl
import dgl.function as fn


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

        self.in_layer = nn.linear(self.in_dim, self.hid_dim, bias=True)
        self.hid_layer = nn.linear(self.hid_dim, hid_dim, bias=True)
        self.out_layer = nn.linear(self.hid_dim, self.out_dim, bias=False)

    def foward(self, graph, n_feat, edge_weights):

        graph.ndata['h'] = n_feat

        graph.apply_edges(fn.copy_u('h', 'm'))
        graph.edata['h'] = graph.edata['m'] * edge_weights
        graph.update_all(fn.copy_e('h', 'm'), fn.sum('m', 'h'))

        graph.ndata['h'] = graph.ndata['h'] + self.in_layer(graph.ndata['h'])

