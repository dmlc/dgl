"""IGMC modules"""

import torch as th 
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import vector_norm

from dgl.nn.pytorch import RelGraphConv

class IGMC(nn.Module):
    # The GNN model of Inductive Graph-based Matrix Completion. 
    # Use RGCN convolution + center-nodes readout.
    
    def __init__(self, in_feats, latent_dim=(32, 32, 32, 32),
                num_relations=5, num_bases=4, edge_dropout=0.2):
        super(IGMC, self).__init__()

        self.edge_dropout = edge_dropout
        self.in_feats = in_feats
        self.num_bases = num_bases

        self.convs = th.nn.ModuleList()
        self.convs.append(RelGraphConv(in_feats, latent_dim[0], num_relations, regularizer='basis',
                                num_bases=num_bases))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(RelGraphConv(latent_dim[i], latent_dim[i+1], num_relations, regularizer='basis',
                                    num_bases=num_bases))
        
        self.lin1 = nn.Linear(2 * sum(latent_dim), 128)
        self.lin2 = nn.Linear(128, 1)

    

    def forward(self, block):
        block = edge_drop(block, self.edge_dropout, self.training)
        concat_states = []
        arr_loss = 0.0
        x = block.ndata['nlabel']
        for conv in self.convs:
            # edge mask zero denotes the edge dropped
            x = th.tanh(conv(block, x, block.edata['etype'], 
                             norm=block.edata['edge_mask'].unsqueeze(1)))
            concat_states.append(x)
            arr_loss += adj_rating_reg(conv, self.training)
        concat_states = th.cat(concat_states, 1)
        
        users = block.ndata['nlabel'][:, 0] == 1
        items = block.ndata['nlabel'][:, 1] == 1
        x = th.cat([concat_states[users], concat_states[items]], 1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x[:, 0], arr_loss

    def __repr__(self):
        return self.__class__.__name__
    
def adj_rating_reg(conv, training):
    if training:
        coeff = conv.linear_r.coeff[:-1] - conv.linear_r.coeff[1:]
        W = conv.linear_r.W.reshape(coeff.shape[1], -1)
        W = th.matmul(coeff, W)
        arr_loss = (vector_norm(W, dim=1) ** 2).sum()
        return arr_loss
    else:
        return 0.0

def edge_drop(graph, edge_dropout=0.2, training=True):
    assert (edge_dropout >= 0.0) and (edge_dropout <= 1.0), 'Invalid dropout rate.'

    if not training:
        return graph

    # set edge mask to zero in directional mode
    src, _ = graph.edges()
    to_drop = src.new_full((graph.number_of_edges(), ), edge_dropout, dtype=th.float)
    to_drop = th.bernoulli(to_drop).to(th.bool)
    graph.edata['edge_mask'][to_drop] = 0

    return graph
