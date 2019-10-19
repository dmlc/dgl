import math

import dgl.function as fn
import torch
import torch.nn as nn

class GCNLayerSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True,
                 use_pp=False,
                 use_lynorm=True):
        super(GCNLayerSAGE, self).__init__()
        # The input feature size gets doubled as we concatenated the original
        # features with the new features.
        self.weight = nn.Parameter(torch.Tensor(2 * in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        self.use_pp = use_pp
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        if use_lynorm:
            self.lynorm = nn.LayerNorm(out_feats, elementwise_affine=True)
        else:
            self.lynorm = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g):
        h = g.ndata['h']

        if not self.use_pp or not self.training:
            norm = self.get_norm(g)
            g.ndata['h'] = h
            g.update_all(fn.copy_src(src='h', out='m'),
                         fn.sum(msg='m', out='h'))
            ah = g.ndata.pop('h')
            h = self.concat(h, ah, norm)

        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)
        if self.bias is not None:
            h = h + self.bias
        h = self.lynorm(h)
        if self.activation:
            h = self.activation(h)
        return h

    def concat(self, h, ah, norm):
        ah = ah * norm
        h = torch.cat((h, ah), dim=1)
        return h

    def get_norm(self, g):
        norm = 1. / g.in_degrees().float().unsqueeze(1)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.weight.device)
        return norm

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 use_pp):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GCNLayerSAGE(in_feats, n_hidden, activation=activation,
                                        dropout=dropout, use_pp=use_pp, use_lynorm=True))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GCNLayerSAGE(n_hidden, n_hidden, activation=activation, dropout=dropout,
                             use_pp=False, use_lynorm=True))
        # output layer
        self.layers.append(GCNLayerSAGE(n_hidden, n_classes, activation=None,
                                        dropout=dropout, use_pp=False, use_lynorm=False))

    def forward(self, g):
        g.ndata['h'] = g.ndata['features']
        for layer in self.layers:
            g.ndata['h'] = layer(g)
        h = g.ndata.pop('h')
        return h
