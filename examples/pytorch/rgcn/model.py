import torch
import torch.nn as nn
import numpy as np

class BaseRGCN(nn.Module):
    def __init__(self, g, h_dim, out_dim, num_rels, num_bases=-1, num_hidden_layers=1, dropout=0, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.g = g
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        raise NotImplementedError

    def build_input_layer(self):
        return None

    def build_hidden_layer(self):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self):
        if self.features is not None:
            self.g.set_n_repr(self.features)
        for layer in self.layers:
            layer(self.g, self.subgraphs)
        return self.g.pop_n_repr()

