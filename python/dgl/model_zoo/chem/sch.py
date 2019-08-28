# -*- coding:utf-8 -*-
# pylint: disable=C0103, C0111, W0621
"""Implementation of SchNet model."""
import torch as th
import torch.nn as nn

from .layers import AtomEmbedding, Interaction, ShiftSoftplus, RBFLayer
from ...batched_graph import sum_nodes


class SchNetModel(nn.Module):
    """
    `SchNet: A continuous-filter convolutional neural network for modeling
    quantum interactions. (NIPS'2017) <https://arxiv.org/abs/1706.08566>`__

    Parameters
    ----------
    dim : int
        Dimension of features, default to be 64
    cutoff : float
        Radius cutoff for RBF, default to be 5.0
    output_dim : int
        Dimension of prediction, default to be 1
    width : int
        Width in RBF, default to 1
    n_conv : int
        Number of conv (interaction) layers, default to be 1
    norm : bool
        Whether to normalize the output atom representations, default to be False.
    atom_ref : Atom embeddings or None
        If None, random representation initialization will be used. Otherwise,
        they will be used to initialize atom representations. Default to be None.
    pre_train : Atom embeddings or None
        If None, random representation initialization will be used. Otherwise,
        they will be used to initialize atom representations. Default to be None.
    """
    def __init__(self,
                 dim=64,
                 cutoff=5.0,
                 output_dim=1,
                 width=1,
                 n_conv=3,
                 norm=False,
                 atom_ref=None,
                 pre_train=None):
        super().__init__()
        self.name = "SchNet"
        self._dim = dim
        self.cutoff = cutoff
        self.width = width
        self.n_conv = n_conv
        self.atom_ref = atom_ref
        self.norm = norm
        self.activation = ShiftSoftplus()

        if atom_ref is not None:
            self.e0 = AtomEmbedding(1, pre_train=atom_ref)
        if pre_train is None:
            self.embedding_layer = AtomEmbedding(dim)
        else:
            self.embedding_layer = AtomEmbedding(pre_train=pre_train)
        self.rbf_layer = RBFLayer(0, cutoff, width)
        self.conv_layers = nn.ModuleList(
            [Interaction(self.rbf_layer._fan_out, dim) for i in range(n_conv)])

        self.atom_dense_layer1 = nn.Linear(dim, 64)
        self.atom_dense_layer2 = nn.Linear(64, output_dim)

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
        self.mean_per_atom = th.tensor(mean, device=device)
        self.std_per_atom = th.tensor(std, device=device)

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
        self.embedding_layer(g)
        if self.atom_ref is not None:
            self.e0(g, "e0")
        self.rbf_layer(g)
        for idx in range(self.n_conv):
            self.conv_layers[idx](g)

        atom = self.atom_dense_layer1(g.ndata["node"])
        atom = self.activation(atom)
        res = self.atom_dense_layer2(atom)
        g.ndata["res"] = res

        if self.atom_ref is not None:
            g.ndata["res"] = g.ndata["res"] + g.ndata["e0"]

        if self.norm:
            g.ndata["res"] = g.ndata["res"] * self.std_per_atom + self.mean_per_atom
        res = sum_nodes(g, "res")
        return res
