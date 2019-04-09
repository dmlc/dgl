"""
APPNP implementation in DGL.
References
----------
Paper: https://arxiv.org/abs/1810.05997
Author's code: https://github.com/klicperajo/ppnp
"""

import torch.nn as nn
import dgl.function as fn


class APPNP(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 hiddens,
                 n_classes,
                 activation,
                 dropout,
                 alpha,
                 k):
        super(APPNP, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g
        # input layer
        self.layers.append(nn.Linear(in_feats, hiddens[0]))
        # hidden layers
        for i in range(1, len(hiddens)):
            self.layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))
        # output layer
        self.layers.append(nn.Linear(hiddens[-1], n_classes))
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.K = k
        self.alpha = alpha

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features):
        # prediction step
        h = features
        if self.dropout:
            h = self.dropout(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        if self.dropout:
            h = self.layers[-1](self.dropout(h))
        # propagation step without dropout on adjacency matrices
        self.cached_h = h
        for _ in range(self.K):
            # normalization by square root of src degree
            h = h * self.g.ndata['norm']
            self.g.ndata['h'] = h
            self.g.update_all(fn.copy_src(src='h', out='m'),
                              fn.sum(msg='m', out='h'))
            h = self.g.ndata.pop('h')
            # normalization by square root of dst degree
            h = h * self.g.ndata['norm']
            # update h using teleport probability alpha
            h = h * (1 - self.alpha) + self.cached_h * self.alpha

        return h
