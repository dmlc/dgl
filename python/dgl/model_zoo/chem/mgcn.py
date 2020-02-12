# -*- coding:utf-8 -*-
# pylint: disable=C0103, C0111, W0621
"""Implementation  of MGCN model"""
import torch
import torch.nn as nn

from .layers import AtomEmbedding, RBFLayer, EdgeEmbedding, \
    MultiLevelInteraction
from ...nn.pytorch import SumPooling
from ...contrib.deprecation import deprecated


class MGCNModel(nn.Module):
    """
    `Molecular Property Prediction: A Multilevel
    Quantum Interactions Modeling Perspective <https://arxiv.org/abs/1906.11081>`__

    Parameters
    ----------
    dim : int
        Size for embeddings, default to be 128.
    width : int
        Width in the RBF layer, default to be 1.
    cutoff : float
        The maximum distance between nodes, default to be 5.0.
    edge_dim : int
        Size for edge embedding, default to be 128.
    out_put_dim: int
        Number of target properties to predict, default to be 1.
    n_conv : int
        Number of convolutional layers, default to be 3.
    norm : bool
        Whether to perform normalization, default to be False.
    atom_ref : Atom embeddings or None
        If None, random representation initialization will be used. Otherwise,
        they will be used to initialize atom representations. Default to be None.
    pre_train : Atom embeddings or None
        If None, random representation initialization will be used. Otherwise,
        they will be used to initialize atom representations. Default to be None.
    """
    @deprecated('Import MGCNPredictor from dgllife.model instead.', 'class')
    def __init__(self,
                 dim=128,
                 width=1,
                 cutoff=5.0,
                 edge_dim=128,
                 output_dim=1,
                 n_conv=3,
                 norm=False,
                 atom_ref=None,
                 pre_train=None):
        super(MGCNModel, self).__init__()

        self._dim = dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.cutoff = cutoff
        self.width = width
        self.n_conv = n_conv
        self.atom_ref = atom_ref
        self.norm = norm

        if pre_train is None:
            self.embedding_layer = AtomEmbedding(dim)
        else:
            self.embedding_layer = AtomEmbedding(pre_train=pre_train)
        self.rbf_layer = RBFLayer(0, cutoff, width)
        self.edge_embedding_layer = EdgeEmbedding(dim=edge_dim)

        if atom_ref is not None:
            self.e0 = AtomEmbedding(1, pre_train=atom_ref)

        self.conv_layers = nn.ModuleList([
            MultiLevelInteraction(self.rbf_layer._fan_out, dim)
            for i in range(n_conv)
        ])

        self.out_project = nn.Sequential(
            nn.Linear(dim * (self.n_conv + 1), 64),
            nn.Softplus(beta=1, threshold=20),
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
        e = self.edge_embedding_layer(g, atom_types)
        rbf_out = self.rbf_layer(edge_distances)

        all_layer_h = [h]
        for idx in range(self.n_conv):
            h, e = self.conv_layers[idx](g, h, e, rbf_out)
            all_layer_h.append(h)

        # concat multilevel representations
        h = torch.cat(all_layer_h, dim=1)
        h = self.out_project(h)

        if self.atom_ref is not None:
            h_ref = self.e0(atom_types)
            h = h + h_ref

        if self.norm:
            h = h * self.std_per_node + self.mean_per_node

        return self.readout(g, h)
