"""NN modules"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import dgl.function as fn
import dgl.nn.pytorch as dglnn

from utils import get_activation

class IGMC(torch.nn.Module):
    # The GNN model of Inductive Graph-based Matrix Completion. Use RGCN convolution + center-nodes readout.
    def __init__(self, in_dim, gconv=dglnn.RelGraphConv, latent_dim=[32, 32, 32, 32], num_relations=5, num_bases=2, regression=False, adj_dropout=0.2, force_undirected=False, side_features=False, n_side_features=0, multiply_by=1):
        super(IGMC, self).__init__()
        self.multiply_by = multiply_by
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(in_dim, latent_dim[0], num_relations, num_bases=num_bases))
        self.regression = regression
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], num_relations, num_bases=num_bases))
        self.lin1 = nn.Linear(2*sum(latent_dim), 128)
        self.side_features = side_features
        if side_features:
            self.lin1 = nn.Linear(2*sum(latent_dim)+n_side_features, 128)
        if self.regression:
            self.lin2 = nn.Linear(128, 1)
        else:
            assert False
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__

    def forward(self, g):
        # Don't drop for now
        '''
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(edge_index, edge_type, p=self.adj_dropout, force_undirected=self.force_undirected, num_nodes=len(x), training=self.training)
        '''
        concat_states = []
        x = g.ndata['x']
        for conv in self.convs:
            x = torch.tanh(conv(g, x, g.edata['etype']))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        users = g.ndata['x'][:, 0] == 1
        items = g.ndata['x'][:, 1] == 1
        x = torch.cat([concat_states[users], concat_states[items]], 1)
        if self.side_features:
            x = torch.cat([x, data.u_feature, data.v_feature], 1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0] * self.multiply_by
        else:
            assert False
