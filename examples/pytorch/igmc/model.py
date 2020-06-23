"""IGMC modules"""

import torch as th 
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import RelGraphConv

class IGMC(nn.Module):
    # The GNN model of Inductive Graph-based Matrix Completion. 
    # Use RGCN convolution + center-nodes readout.
    
    def __init__(self, in_feats, gconv=RelGraphConv, latent_dim=[32, 32, 32, 32], 
                num_relations=5, num_bases=2, regression=False, edge_dropout=0.2, 
                force_undirected=False, side_features=False, n_side_features=0, 
                multiply_by=1):
        super(IGMC, self).__init__()

        self.regression = regression
        self.edge_dropout = edge_dropout
        self.force_undirected = force_undirected
        self.side_features = side_features
        self.multiply_by = multiply_by

        self.convs = th.nn.ModuleList()
        self.convs.append(gconv(in_feats, latent_dim[0], num_relations, num_bases=num_bases, self_loop=True))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], num_relations, num_bases=num_bases, self_loop=True))
        
        self.lin1 = nn.Linear(2 * sum(latent_dim), 128)
        if side_features:
            self.lin1 = nn.Linear(2 * sum(latent_dim) + n_side_features, 128)
        if self.regression:
            self.lin2 = nn.Linear(128, 1)
        else:
            assert False
            # self.lin2 = nn.Linear(128, n_classes)
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, block):
        block = edge_drop(block, self.edge_dropout, self.force_undirected, self.training)

        concat_states = []
        x = block.ndata['x']
        for conv in self.convs:
            x = th.tanh(conv(block, x, block.edata['etype']))
            concat_states.append(x)
        concat_states = th.cat(concat_states, 1)
        
        users = block.ndata['x'][:, 0] == 1
        items = block.ndata['x'][:, 1] == 1
        x = th.cat([concat_states[users], concat_states[items]], 1)
        # if self.side_features:
        #     x = th.cat([x, data.u_feature, data.v_feature], 1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0] * self.multiply_by
        else:
            assert False
            # return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

# XXX [zhoujf] slower than dropout_link fn that directly manipulates on array
def edge_drop(dgl_graph, edge_dropout=0.2, force_undirected=False, training=True):
    assert edge_dropout >= 0.0 and edge_dropout <= 1.0, 'Invalid dropout rate.'

    if not training:
        return dgl_graph

    # cal dropout mask
    src, dst = dgl_graph.edges()
    n_edges = dgl_graph.number_of_edges() // 2 if force_undirected else dgl_graph.number_of_edges()

    mask = src.new_full((n_edges, ), 1 - edge_dropout, dtype=th.float)
    mask = th.bernoulli(mask).to(th.bool)
    if force_undirected:
        mask = th.concat([mask, mask], axis=0)
        
    src, dst = src[mask], dst[mask]
    edges_to_keep = dgl_graph.edge_ids(src, dst)
    # the pair of first user and movie nodes may be isolated
    subgraph = dgl_graph.edge_subgraph(edges_to_keep, preserve_nodes=True)

    # restore node and edge features
    for k in dgl_graph.ndata.keys():
        subgraph.ndata[k] = dgl_graph.ndata[k][subgraph.parent_nid]
    for k in dgl_graph.edata.keys():
        subgraph.edata[k] = dgl_graph.edata[k][subgraph.parent_eid]

    return subgraph
