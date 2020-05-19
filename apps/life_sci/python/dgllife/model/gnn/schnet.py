# -*- coding:utf-8 -*-
# pylint: disable=C0103, C0111, W0621, W0221, E1102, E1101
"""SchNet"""
import numpy as np
import torch
import torch.nn as nn

from dgl.nn.pytorch import CFConv

__all__ = ['SchNetGNN']

class RBFExpansion(nn.Module):
    r"""Expand distances between nodes by radial basis functions.

    .. math::
        \exp(- \gamma * ||d - \mu||^2)

    where :math:`d` is the distance between two nodes and :math:`\mu` helps centralizes
    the distances. We use multiple centers evenly distributed in the range of
    :math:`[\text{low}, \text{high}]` with the difference between two adjacent centers
    being :math:`gap`.

    The number of centers is decided by :math:`(\text{high} - \text{low}) / \text{gap}`.
    Choosing fewer centers corresponds to reducing the resolution of the filter.

    Parameters
    ----------
    low : float
        Smallest center. Default to 0.
    high : float
        Largest center. Default to 30.
    gap : float
        Difference between two adjacent centers. :math:`\gamma` will be computed as the
        reciprocal of gap. Default to 0.1.
    """
    def __init__(self, low=0., high=30., gap=0.1):
        super(RBFExpansion, self).__init__()

        num_centers = int(np.ceil((high - low) / gap))
        centers = np.linspace(low, high, num_centers)
        self.centers = nn.Parameter(torch.tensor(centers).float(), requires_grad=False)
        self.gamma = 1 / gap

    def forward(self, edge_dists):
        """Expand distances.

        Parameters
        ----------
        edge_dists : float32 tensor of shape (E, 1)
            Distances between end nodes of edges, E for the number of edges.

        Returns
        -------
        float32 tensor of shape (E, len(self.centers))
            Expanded distances.
        """
        radial = edge_dists - self.centers
        coef = - self.gamma
        return torch.exp(coef * (radial ** 2))

class Interaction(nn.Module):
    """Building block for SchNet.

    SchNet is introduced in `SchNet: A continuous-filter convolutional neural network for
    modeling quantum interactions <https://arxiv.org/abs/1706.08566>`__.

    This layer combines node and edge features in message passing and updates node
    representations.

    Parameters
    ----------
    node_feats : int
        Size for the input and output node features.
    edge_in_feats : int
        Size for the input edge features.
    hidden_feats : int
        Size for hidden representations.
    """
    def __init__(self, node_feats, edge_in_feats, hidden_feats):
        super(Interaction, self).__init__()

        self.conv = CFConv(node_feats, edge_in_feats, hidden_feats, node_feats)
        self.project_out = nn.Linear(node_feats, node_feats)

    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feats)
            Input node features, V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features, E for the number of edges.

        Returns
        -------
        float32 tensor of shape (V, node_feats)
            Updated node representations.
        """
        node_feats = self.conv(g, node_feats, edge_feats)
        return self.project_out(node_feats)

class SchNetGNN(nn.Module):
    """SchNet.

    SchNet is introduced in `SchNet: A continuous-filter convolutional neural network for
    modeling quantum interactions <https://arxiv.org/abs/1706.08566>`__.

    This class performs message passing in SchNet and returns the updated node representations.

    Parameters
    ----------
    node_feats : int
        Size for node representations to learn. Default to 64.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of hidden representations for the i-th interaction
        layer. ``len(hidden_feats)`` equals the number of interaction layers.
        Default to ``[64, 64, 64]``.
    num_node_types : int
        Number of node types to embed. Default to 100.
    cutoff : float
        Largest center in RBF expansion. Default to 30.
    gap : float
        Difference between two adjacent centers in RBF expansion. Default to 0.1.
    """
    def __init__(self, node_feats=64, hidden_feats=None, num_node_types=100, cutoff=30., gap=0.1):
        super(SchNetGNN, self).__init__()

        if hidden_feats is None:
            hidden_feats = [64, 64, 64]

        self.embed = nn.Embedding(num_node_types, node_feats)
        self.rbf = RBFExpansion(high=cutoff, gap=gap)

        n_layers = len(hidden_feats)
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(
                Interaction(node_feats, len(self.rbf.centers), hidden_feats[i]))

    def forward(self, g, node_types, edge_dists):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_types : int64 tensor of shape (V)
            Node types to embed, V for the number of nodes.
        edge_dists : float32 tensor of shape (E, 1)
            Distances between end nodes of edges, E for the number of edges.

        Returns
        -------
        node_feats : float32 tensor of shape (V, node_feats)
            Updated node representations.
        """
        node_feats = self.embed(node_types)
        expanded_dists = self.rbf(edge_dists)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats, expanded_dists)
        return node_feats
