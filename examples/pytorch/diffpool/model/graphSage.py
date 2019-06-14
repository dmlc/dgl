import torch
import torch.nn as nn

import dgl.function as fn

from .aggregator import MaxPoolAggregator, MeanAggregator, LSTMAggregator
from .bundler import Bundler


class GraphSageLayer(nn.Module):
    """
    GraphSage layer in Inductive learning paper by hamilton
    Here, graphsage layer is a reduced function in DGL framework
    """
    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type, bias=True):
        super(GraphSageLayer, self).__init__()
        self.bundler = Bundler(in_feats, out_feats, activation, dropout,
                               bias=True)
        self.dropout = nn.Dropout(p=dropout)

        if aggregator_type == "maxpool":
            self.aggregator = MaxPoolAggregator(in_feats, in_feats,
                                                activation, bias)
        elif aggregator_type == "lstm":
            self.aggregator = LSTMAggregator(in_feats, in_feats)
        else:
            self.aggregator = MeanAggregator()

    def forward(self, g, h):
        h = self.dropout(h)
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'), self.aggregator,
                     self.bundler)
        h = g.ndata.pop('h')
        return h


class GraphSage(nn.Module):
    """
    Grahpsage network that concatenate several graphsage layer
    """
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout, aggregator_type):
        super(GraphSage, self).__init__()
        self.layers = nn.ModuleList()

        #input layer
        self.layers.append(GraphSageLayer(in_feats, n_hidden, activation, dropout,
                                          aggregator_type))
        # hidden layers
        for _ in range(n_layers -1):
            self.layers.append(GraphSageLayer(n_hidden, n_hidden, activation,
                                              dropout, aggregator_type))
        #output layer
        self.layers.append(GraphSageLayer(n_hidden, n_classes, None,
                                          dropout, aggregator_type))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h
