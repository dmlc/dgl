import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl import DGLGraph

class MWEConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 bias=True,
                 num_channels=8,
                 aggr_mode='sum'):
        super(MWEConv, self).__init__()
        self.num_channels = num_channels
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats, num_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats, num_channels))
        else:
            self.bias = None
        self.reset_parameters()
        self.activation = activation

        if (aggr_mode == 'concat'):
            self.aggr_mode = 'concat'
            self.final = nn.Linear(out_feats * self.num_channels, out_feats)
        elif (aggr_mode == 'sum'):
            self.aggr_mode = 'sum'
            self.final = nn.Linear(out_feats, out_feats)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, node_state_prev):
        node_state = node_state_prev

        # if self.dropout:
        #     node_states = self.dropout(node_state)

        g = g.local_var()

        new_node_states = []

        ## perform weighted convolution for every channel of edge weight
        for c in range(self.num_channels):
            node_state_c = node_state
            if self._out_feats < self._in_feats:
                g.ndata['feat_' + str(c)] = torch.mm(node_state_c, self.weight[:, :, c])
            else:
                g.ndata['feat_' + str(c)] = node_state_c
            g.update_all(fn.src_mul_edge('feat_' + str(c), 'feat_' + str(c), 'm'), fn.sum('m', 'feat_' + str(c) + '_new'))
            node_state_c = g.ndata.pop('feat_' + str(c) + '_new')
            if self._out_feats >= self._in_feats:
                node_state_c = torch.mm(node_state_c, self.weight[:, :, c])          
            if self.bias is not None:
                node_state_c = node_state_c + self.bias[:, c]
            node_state_c = self.activation(node_state_c)   
            new_node_states.append(node_state_c) 
        if (self.aggr_mode == 'sum'):
            node_states = torch.stack(new_node_states, dim=1).sum(1)
        elif (self.aggr_mode == 'concat'):
            node_states = torch.cat(new_node_states, dim=1)

        node_states = self.final(node_states)

        return node_states


class MWE_GCN(nn.Module):
    def __init__(self,
                 n_input,
                 n_hidden,
                 n_output,
                 n_layers,
                 activation,
                 dropout,
                 aggr_mode='sum',
                 device='cpu'):
        super(MWE_GCN, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.layers = nn.ModuleList()

        self.layers.append(MWEConv(n_input, n_hidden, activation=activation, \
            aggr_mode=aggr_mode))
        for i in range(n_layers - 1):
            self.layers.append(MWEConv(n_hidden, n_hidden, activation=activation, \
                aggr_mode=aggr_mode))

        self.pred_out = nn.Linear(n_hidden, n_output)
        self.device = device

    def forward(self, g, node_state=None):
        node_state = torch.ones(g.number_of_nodes(), 1).float().to(self.device)

        for layer in self.layers:
            node_state = F.dropout(node_state, p=self.dropout, training=self.training)
            node_state = layer(g, node_state)
            node_state = self.activation(node_state)              

        out = self.pred_out(node_state)
        return out


class MWE_DGCN(nn.Module):
    def __init__(self,
                 n_input,
                 n_hidden,
                 n_output,
                 n_layers,
                 activation,
                 dropout,
                 residual=False,
                 aggr_mode='sum',
                 device='cpu'):
        super(MWE_DGCN, self).__init__()
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
        self.residual = residual

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        self.layers.append(MWEConv(n_input, n_hidden, activation=activation, \
            aggr_mode=aggr_mode))
        
        for i in range(n_layers - 1):
            self.layers.append(MWEConv(n_hidden, n_hidden, activation=activation, \
                aggr_mode=aggr_mode))

        for i in range(n_layers):
            self.layer_norms.append(nn.LayerNorm(n_hidden, elementwise_affine=True))

        self.pred_out = nn.Linear(n_hidden, n_output)
        self.device = device


    def forward(self, g, node_state=None):
        node_state = torch.ones(g.number_of_nodes(), 1).float().to(self.device)

        node_state = self.layers[0](g, node_state)

        for layer in range(1, self.n_layers):
            node_state_new = self.layer_norms[layer-1](node_state)
            node_state_new = self.activation(node_state_new)
            node_state_new = F.dropout(node_state_new, p=self.dropout, training=self.training)

            if (self.residual == 'true'):
                node_state = node_state + self.layers[layer](g, node_state_new)
            else:
                node_state = self.layers[layer](g, node_state_new)

        node_state = self.layer_norms[self.n_layers-1](node_state)
        node_state = self.activation(node_state)
        node_state = F.dropout(node_state, p=self.dropout, training=self.training)

        out = self.pred_out(node_state)

        return out


