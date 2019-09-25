# -*- coding: utf-8 -*-
# pylint: disable=C0103, E1101, C0111
"""
The implementation of neural network layers used in SchNet and MGCN.
"""
import torch
import torch.nn as nn
from torch.nn import Softplus
import numpy as np

from ... import function as fn

class AtomEmbedding(nn.Module):
    """
    Convert the atom(node) list to atom embeddings.
    The atoms with the same element share the same initial embedding.

    Parameters
    ----------
    dim : int
        Size of embeddings, default to be 128.
    type_num : int
        The largest atomic number of atoms in the dataset, default to be 100.
    pre_train : None or pre-trained embeddings
        Pre-trained embeddings, default to be None.
    """
    def __init__(self, dim=128, type_num=100, pre_train=None):
        super(AtomEmbedding, self).__init__()

        self._dim = dim
        self._type_num = type_num
        if pre_train is not None:
            self.embedding = nn.Embedding.from_pretrained(pre_train, padding_idx=0)
        else:
            self.embedding = nn.Embedding(type_num, dim, padding_idx=0)

    def forward(self, atom_types):
        """
        Parameters
        ----------
        atom_types : int64 tensor of shape (B1)
            Types for atoms in the graph(s), B1 for the number of atoms.

        Returns
        -------
        float32 tensor of shape (B1, self._dim)
            Atom embeddings.
        """
        return self.embedding(atom_types)

class EdgeEmbedding(nn.Module):
    """
    Module for embedding edges. Edges linking same pairs of atoms share
    the same initial embedding.

    Parameters
    ----------
    dim : int
        Size of embeddings, default to be 128.
    edge_num : int
        Maximum number of edge types allowed, default to be 3000.
    pre_train : Edge embeddings or None
        Pre-trained edge embeddings, default to be None.
    """
    def __init__(self, dim=128, edge_num=3000, pre_train=None):
        super(EdgeEmbedding, self).__init__()

        self._dim = dim
        self._edge_num = edge_num
        if pre_train is not None:
            self.embedding = nn.Embedding.from_pretrained(pre_train, padding_idx=0)
        else:
            self.embedding = nn.Embedding(edge_num, dim, padding_idx=0)

    def generate_edge_type(self, edges):
        """Generate edge type.

        The edge type is based on the type of the src & dst atom.
        Note that directions are not distinguished, e.g. C-O and O-C are the same edge type.

        To map a pair of nodes to one number, we use an unordered pairing function here
        See more detail in this disscussion:
        https://math.stackexchange.com/questions/23503/create-unique-number-from-2-numbers
        Note that the edge_num should be larger than the square of maximum atomic number
        in the dataset.

        Parameters
        ----------
        edges : EdgeBatch
            Edges for deciding types

        Returns
        -------
        dict
            Stores the edge types in "type"
        """
        atom_type_x = edges.src['ntype']
        atom_type_y = edges.dst['ntype']

        return {
            'etype': atom_type_x * atom_type_y + \
                     (torch.abs(atom_type_x - atom_type_y) - 1) ** 2 / 4
        }

    def forward(self, g, atom_types):
        """Compute edge embeddings

        Parameters
        ----------
        g : DGLGraph
            The graph to compute edge embeddings
        atom_types : int64 tensor of shape (B1)
            Types for atoms in the graph(s), B1 for the number of atoms.

        Returns
        -------
        float32 tensor of shape (B2, self._dim)
            Computed edge embeddings
        """
        g = g.local_var()
        g.ndata['ntype'] = atom_types
        g.apply_edges(self.generate_edge_type)
        return self.embedding(g.edata.pop('etype'))

class ShiftSoftplus(nn.Module):
    """
    ShiftSoftplus activation function:
        1/beta * (log(1 + exp**(beta * x)) - log(shift))

    Parameters
    ----------
    beta : int
        Default to be 1.
    shift : int
        Default to be 2.
    threshold : int
        Default to be 20.
    """
    def __init__(self, beta=1, shift=2, threshold=20):
        super(ShiftSoftplus, self).__init__()

        self.shift = shift
        self.softplus = Softplus(beta, threshold)

    def forward(self, x):
        """Applies the activation function"""
        return self.softplus(x) - np.log(float(self.shift))

class RBFLayer(nn.Module):
    """
    Radial basis functions Layer.

    e(d) = exp(- gamma * ||d - mu_k||^2)

    With the default parameters below, we are using a default settings:
    * gamma = 10
    * 0 <= mu_k <= 30 for k=1~300

    Parameters
    ----------
    low : int
        Smallest value to take for mu_k, default to be 0.
    high : int
        Largest value to take for mu_k, default to be 30.
    gap : float
        Difference between two consecutive values for mu_k, default to be 0.1.
    dim : int
        Output size for each center, default to be 1.
    """
    def __init__(self, low=0, high=30, gap=0.1, dim=1):
        super(RBFLayer, self).__init__()

        self._low = low
        self._high = high
        self._dim = dim

        self._n_centers = int(np.ceil((high - low) / gap))
        centers = np.linspace(low, high, self._n_centers)
        self.centers = torch.tensor(centers, dtype=torch.float, requires_grad=False)
        self.centers = nn.Parameter(self.centers, requires_grad=False)
        self._fan_out = self._dim * self._n_centers
        self._gap = centers[1] - centers[0]

    def forward(self, edge_distances):
        """
        Parameters
        ----------
        edge_distances : float32 tensor of shape (B, 1)
            Edge distances, B for the number of edges.

        Returns
        -------
        float32 tensor of shape (B, self._fan_out)
            Computed RBF results
        """
        radial = edge_distances - self.centers
        coef = -1 / self._gap
        return torch.exp(coef * (radial ** 2))

class CFConv(nn.Module):
    """
    The continuous-filter convolution layer in SchNet.

    Parameters
    ----------
    rbf_dim : int
        Dimension of the RBF layer output
    dim : int
        Dimension of output, default to be 64
    act : activation function or None.
        Activation function, default to be shifted softplus
    """
    def __init__(self, rbf_dim, dim=64, act=None):
        super(CFConv, self).__init__()

        self._rbf_dim = rbf_dim
        self._dim = dim

        if act is None:
            activation = nn.Softplus(beta=0.5, threshold=14)
        else:
            activation = act

        self.project = nn.Sequential(
            nn.Linear(self._rbf_dim, self._dim),
            activation,
            nn.Linear(self._dim, self._dim)
        )

    def forward(self, g, node_weight, rbf_out):
        """
        Parameters
        ----------
        g : DGLGraph
            The graph for performing convolution
        node_weight : float32 tensor of shape (B1, D1)
            The weight of nodes in message passing, B1 for number of nodes and
            D1 for node weight size.
        rbf_out : float32 tensor of shape (B2, D2)
            The output of RBFLayer, B2 for number of edges and D2 for rbf out size.
        """
        g = g.local_var()
        e = self.project(rbf_out)
        g.ndata['node_weight'] = node_weight
        g.edata['e'] = e
        g.update_all(fn.u_mul_e('node_weight', 'e', 'm'), fn.sum('m', 'h'))
        return g.ndata.pop('h')

class Interaction(nn.Module):
    """
    The interaction layer in the SchNet model.

    Parameters
    ----------
    rbf_dim : int
        Dimension of the RBF layer output
    dim : int
        Dimension of intermediate representations
    """
    def __init__(self, rbf_dim, dim):
        super(Interaction, self).__init__()

        self._dim = dim
        self.node_layer1 = nn.Linear(dim, dim, bias=False)
        self.cfconv = CFConv(rbf_dim, dim, Softplus(beta=0.5, threshold=14))
        self.node_layer2 = nn.Sequential(
            nn.Linear(dim, dim),
            Softplus(beta=0.5, threshold=14),
            nn.Linear(dim, dim)
        )

    def forward(self, g, n_feat, rbf_out):
        """
        Parameters
        ----------
        g : DGLGraph
            The graph for performing convolution
        n_feat : float32 tensor of shape (B1, D1)
            Node features, B1 for number of nodes and D1 for feature size.
        rbf_out : float32 tensor of shape (B2, D2)
            The output of RBFLayer, B2 for number of edges and D2 for rbf out size.

        Returns
        -------
        float32 tensor of shape (B1, D1)
            Updated node representations
        """
        n_weight = self.node_layer1(n_feat)
        new_n_feat = self.cfconv(g, n_weight, rbf_out)
        new_n_feat = self.node_layer2(new_n_feat)
        return n_feat + new_n_feat

class VEConv(nn.Module):
    """
    The Vertex-Edge convolution layer in MGCN which takes both edge & vertex features
    in consideration.

    Parameters
    ----------
    rbf_dim : int
        Size of the RBF layer output
    dim : int
        Size of intermediate representations, default to be 64.
    update_edge : bool
        Whether to apply a linear layer to update edge representations, default to be True.
    """
    def __init__(self, rbf_dim, dim=64, update_edge=True):
        super(VEConv, self).__init__()

        self._rbf_dim = rbf_dim
        self._dim = dim
        self._update_edge = update_edge

        self.update_rbf = nn.Sequential(
            nn.Linear(self._rbf_dim, self._dim),
            nn.Softplus(beta=0.5, threshold=14),
            nn.Linear(self._dim, self._dim)
        )
        self.update_efeat = nn.Linear(self._dim, self._dim)

    def forward(self, g, n_feat, e_feat, rbf_out):
        """
        Parameters
        ----------
        g : DGLGraph
            The graph for performing convolution
        n_feat : float32 tensor of shape (B1, D1)
            Node features, B1 for number of nodes and D1 for feature size.
        e_feat : float32 tensor of shape (B2, D2)
            Edge features. B2 for number of edges and D2 for
            the edge feature size.
        rbf_out : float32 tensor of shape (B2, D3)
            The output of RBFLayer, B2 for number of edges and D3 for rbf out size.

        Returns
        -------
        n_feat : float32 tensor
            Updated node features.
        e_feat : float32 tensor
            (Potentially updated) edge features
        """
        rbf_out = self.update_rbf(rbf_out)

        if self._update_edge:
            e_feat = self.update_efeat(e_feat)

        g = g.local_var()
        g.ndata.update({'n_feat': n_feat})
        g.edata.update({'rbf_out': rbf_out, 'e_feat': e_feat})
        g.update_all(message_func=[fn.u_mul_e('n_feat', 'rbf_out', 'm_0'),
                                   fn.copy_e('e_feat', 'm_1')],
                     reduce_func=[fn.sum('m_0', 'n_feat_0'),
                                  fn.sum('m_1', 'n_feat_1')])
        n_feat = g.ndata.pop('n_feat_0') + g.ndata.pop('n_feat_1')

        return n_feat, e_feat

class MultiLevelInteraction(nn.Module):
    """
    The multilevel interaction in the MGCN model.

    Parameters
    ----------
    rbf_dim : int
        Dimension of the RBF layer output
    dim : int
        Dimension of intermediate representations
    """
    def __init__(self, rbf_dim, dim):
        super(MultiLevelInteraction, self).__init__()

        self._atom_dim = dim
        self.node_layer1 = nn.Linear(dim, dim, bias=True)
        self.conv_layer = VEConv(rbf_dim, dim)
        self.activation = nn.Softplus(beta=0.5, threshold=14)
        self.edge_layer1 = nn.Linear(dim, dim, bias=True)

        self.node_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Softplus(beta=0.5, threshold=14),
            nn.Linear(dim, dim)
        )

    def forward(self, g, n_feat, e_feat, rbf_out):
        """
        Parameters
        ----------
        g : DGLGraph
            The graph for performing convolution
        n_feat : float32 tensor of shape (B1, D1)
            Node features, B1 for number of nodes and D1 for feature size.
        e_feat : float32 tensor of shape (B2, D2)
            Edge features. B2 for number of edges and D2 for
            the edge feature size.
        rbf_out : float32 tensor of shape (B2, D3)
            The output of RBFLayer, B2 for number of edges and D3 for rbf out size.

        Returns
        -------
        n_feat : float32 tensor
            Updated node representations
        e_feat : float32 tensor
            Updated edge representations
        """
        new_n_feat = self.node_layer1(n_feat)
        new_n_feat, e_feat = self.conv_layer(g, new_n_feat, e_feat, rbf_out)
        new_n_feat = self.node_out(new_n_feat)
        n_feat = n_feat + new_n_feat

        e_feat = self.activation(self.edge_layer1(e_feat))

        return n_feat, e_feat
