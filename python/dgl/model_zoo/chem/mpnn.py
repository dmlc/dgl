#!/usr/bin/env python
# coding: utf-8
# pylint: disable=C0103, C0111, E1101, W0612
"""Implementation of MPNN model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from ... import function as fn
from ...nn.pytorch import Set2Set


class NNConvLayer(nn.Module):
    """
    MPNN Conv Layer from Section 5 of
    `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`__

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    edge_net : Module processing edge information
    root_weight : bool
        Whether to add the root node feature to output
    bias : bool
        Whether to add bias to the output
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_net,
                 root_weight=True,
                 bias=True):
        super(NNConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_net = edge_net

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize model parameters"""
        if self.root is not None:
            nn.init.xavier_normal_(self.root.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.zero_()
        for m in self.edge_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.414)

    def message(self, edges):
        """Function for computing messages from source nodes

        Parameters
        ----------
        edges : EdgeBatch
            Edges over which we want to send messages

        Returns
        -------
        dict
            Stores message in key 'm'
        """
        return {
            'm':
            torch.matmul(edges.src['h'].unsqueeze(1),
                         edges.data['w']).squeeze(1)
        }

    def apply_node_func(self, nodes):
        """Function for updating node features directly

        Parameters
        ----------
        nodes : NodeBatch

        Returns
        -------
        dict
            Stores updated node features in 'h'
        """
        aggr_out = nodes.data['aggr_out']
        if self.root is not None:
            aggr_out = torch.mm(nodes.data['h'], self.root) + aggr_out

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return {'h': aggr_out}

    def forward(self, g, h, e):
        """Propagate messages and aggregate results for updating
        atom representations

        Parameters
        ----------
        g : DGLGraph
            DGLgraph(s) for molecules
        h : tensor
            Input atom representations
        e : tensor
            Input bond representations

        Returns
        -------
        tensor
            Aggregated atom information
        """
        h = h.unsqueeze(-1) if h.dim() == 1 else h
        e = e.unsqueeze(-1) if e.dim() == 1 else e

        g.ndata['h'] = h
        g.edata['w'] = self.edge_net(e).view(-1, self.in_channels,
                                             self.out_channels)
        g.update_all(self.message, fn.sum("m", "aggr_out"),
                     self.apply_node_func)
        return g.ndata.pop('h')


class MPNNModel(nn.Module):
    """
    MPNN from
    `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`__

    Parameters
    ----------
    node_input_dim : int
        Dimension of input node feature, default to be 15.
    edge_input_dim : int
        Dimension of input edge feature, default to be 15.
    output_dim : int
        Dimension of prediction, default to be 12.
    node_hidden_dim : int
        Dimension of node feature in hidden layers, default to be 64.
    edge_hidden_dim : int
        Dimension of edge feature in hidden layers, default to be 128.
    num_step_message_passing : int
        Number of message passing steps, default to be 6.
    num_step_set2set : int
        Number of set2set steps
    num_layer_set2set : int
        Number of set2set layers
    """
    def __init__(self,
                 node_input_dim=15,
                 edge_input_dim=5,
                 output_dim=12,
                 node_hidden_dim=64,
                 edge_hidden_dim=128,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3):
        super(MPNNModel, self).__init__()

        self.name = "MPNN"
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
        self.conv = NNConvLayer(in_channels=node_hidden_dim,
                                out_channels=node_hidden_dim,
                                edge_net=edge_network,
                                root_weight=False)
        self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)

        self.set2set = Set2Set(node_hidden_dim, num_step_set2set, num_layer_set2set)
        self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = nn.Linear(node_hidden_dim, output_dim)

    def forward(self, g):
        """Predict molecule labels

        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph for molecule(s)

        Returns
        -------
        res : Predicted labels
        """
        h = g.ndata['n_feat']
        out = F.relu(self.lin0(h))
        h = out.unsqueeze(0)

        for i in range(self.num_step_message_passing):
            m = F.relu(self.conv(g, out, g.edata['e_feat']))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(g, out)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out
