import torch
import torch.nn as nn
import numpy as np

class BaseRGCN(nn.Module):
    def __init__(self, g, h_dim, out_dim, relations, num_bases=-1, num_hidden_layers=1, dropout=0, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.g = g
        self.dropout = dropout
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = relations.shape[1]
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_cuda = use_cuda

        # FIXME: is this correct for block decompose case?
        assert self.num_bases <= self.num_rels

        # generate subgraphs
        self.build_subgraph_per_relation(relations)

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def build_subgraph_per_relation(self, relations):
        self.subgraphs = []
        src, dst = np.transpose(np.array(self.g.edge_list))
        for rel in range(self.num_rels):
            sub_rel = relations[:, rel]
            if np.count_nonzero(sub_rel) == 0:
                # skip relations with no edges
                continue
            sub_eid = sub_rel > 0
            u = src[sub_eid]
            v = dst[sub_eid]
            sub_rel = sub_rel[sub_eid]
            subgrh = self.g.edge_subgraph(u, v)
            edge_repr = torch.from_numpy(sub_rel).view(-1, 1)
            if self.use_cuda:
                edge_repr = edge_repr.cuda()
            subgrh.set_e_repr(edge_repr)
            self.subgraphs.append(subgrh)

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

