# -*- coding:utf-8 -*-
# pylint: disable=C0103, C0111, W0621
"""Implementation of SchNet model."""
import torch
import torch.nn as nn

from .layers import AtomEmbedding, Interaction, ShiftSoftplus, RBFLayer
from ...contrib.deprecation import deprecated
from ...nn.pytorch import SumPooling


class SchNet(nn.Module):
    """
    `SchNet: A continuous-filter convolutional neural network for modeling
    quantum interactions. (NIPS'2017) <https://arxiv.org/abs/1706.08566>`__

    Parameters
    ----------
    dim : int
        Size for atom embeddings, default to be 64.
    cutoff : float
        Radius cutoff for RBF, default to be 5.0.
    output_dim : int
        Number of target properties to predict, default to be 1.
    width : int
        Width in RBF, default to 1.
    n_conv : int
        Number of conv (interaction) layers, default to be 1.
    norm : bool
        Whether to normalize the output atom representations, default to be False.
    atom_ref : Atom embeddings or None
        If None, random representation initialization will be used. Otherwise,
        they will be used to initialize atom representations. Default to be None.
    pre_train : Atom embeddings or None
        If None, random representation initialization will be used. Otherwise,
        they will be used to initialize atom representations. Default to be None.
    """
    @deprecated('Import SchNetPredictor from dgllife.model instead.')
    def __init__(self,
                 dim=64,
                 cutoff=5.0,
                 output_dim=1,
                 width=1,
                 n_conv=3,
                 norm=False,
                 atom_ref=None,
                 pre_train=None):
        super(SchNet, self).__init__()

        self._dim = dim
        self.cutoff = cutoff
        self.width = width
        self.n_conv = n_conv
        self.atom_ref = atom_ref
        self.norm = norm

        if atom_ref is not None:
            self.e0 = AtomEmbedding(1, pre_train=atom_ref)

        if pre_train is None:
            self.embedding_layer = AtomEmbedding(dim)
        else:
            self.embedding_layer = AtomEmbedding(pre_train=pre_train)

        self.rbf_layer = RBFLayer(0, cutoff, width)
        self.conv_layers = nn.ModuleList(
            [Interaction(self.rbf_layer._fan_out, dim) for _ in range(n_conv)])
        self.atom_update = nn.Sequential(
            nn.Linear(dim, 64),
            ShiftSoftplus(),
            nn.Linear(64, output_dim)
        )
        self.readout = SumPooling()

    def set_mean_std(self, mean, std, device="cpu"):
        """Set the mean and std of atom representations for normalization.

        Parameters
        ----------
        mean : list or numpy array
            The mean of labels
        std : list or numpy array
            The std of labels
        device : str or torch.device
            Device for storing the mean and std
        """
        self.mean_per_node = torch.tensor(mean, device=device)
        self.std_per_node = torch.tensor(std, device=device)

    def forward(self, g, atom_types, edge_distances):
        """Predict molecule labels

        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph for molecule(s)
        atom_types : int64 tensor of shape (B1)
            Types for atoms in the graph(s), B1 for the number of atoms.
        edge_distances : float32 tensor of shape (B2, 1)
            Edge distances, B2 for the number of edges.

        Returns
        -------
        prediction : float32 tensor of shape (B, output_dim)
            Model prediction for the batch of graphs, B for the number
            of graphs, output_dim for the prediction size.
        """
        h = self.embedding_layer(atom_types)
        rbf_out = self.rbf_layer(edge_distances)
        for idx in range(self.n_conv):
            h = self.conv_layers[idx](g, h, rbf_out)
        h = self.atom_update(h)

        if self.atom_ref is not None:
            h_ref = self.e0(atom_types)
            h = h + h_ref

        if self.norm:
            h = h * self.std_per_node + self.mean_per_node

        return self.readout(g, h)
