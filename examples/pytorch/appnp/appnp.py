"""
APPNP implementation in DGL.
References
----------
Paper: https://arxiv.org/abs/1810.05997
Author's code: https://github.com/klicperajo/ppnp
"""
import torch
import torch.nn as nn
import dgl.function as fn


class GraphPropagation(nn.Module):
    def __init__(self,
                 g,
                 edge_drop,
                 alpha,
                 k):
        super(GraphPropagation, self).__init__()
        self.g = g
        self.alpha = alpha
        self.k = k
        if edge_drop:
            self.edge_drop = nn.Dropout(edge_drop)
        else:
            self.edge_drop = 0.

    def forward(self, h):
        self.cached_h = h
        for _ in range(self.k):
            # normalization by square root of src degree
            h = h * self.g.ndata['norm']
            self.g.ndata['h'] = h
            if self.edge_drop:
                # performing edge dropout
                ed = self.edge_drop(torch.ones((self.g.number_of_edges(), 1)))
                self.g.edata['d'] = ed
                self.g.update_all(fn.src_mul_edge(src='h', edge='d', out='m'),
                                  fn.sum(msg='m', out='h'))
            else:
                self.g.update_all(fn.copy_src(src='h', out='m'),
                                  fn.sum(msg='m', out='h'))
            h = self.g.ndata.pop('h')
            # normalization by square root of dst degree
            h = h * self.g.ndata['norm']
            # update h using teleport probability alpha
            h = h * (1 - self.alpha) + self.cached_h * self.alpha
        return h


class APPNP(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 hiddens,
                 n_classes,
                 activation,
                 feat_drop,
                 edge_drop,
                 alpha,
                 k):
        super(APPNP, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, hiddens[0]))
        # hidden layers
        for i in range(1, len(hiddens)):
            self.layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))
        # output layer
        self.layers.append(nn.Linear(hiddens[-1], n_classes))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = GraphPropagation(g, edge_drop, alpha, k)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(h)
        return h
