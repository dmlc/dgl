# -*- coding: utf-8 -*-
# pylint: disable=C0103, E1101, C0111
"""
The implementation of neural network layers used in SchNet and MGCN.
"""
import torch as th
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
        Dim of embeddings, default to be 128.
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

    def forward(self, g, p_name="node"):
        """
        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph(s)
        p_name : str
            Name for storing atom embeddings
        """
        atom_list = g.ndata["node_type"]
        g.ndata[p_name] = self.embedding(atom_list)
        return g.ndata[p_name]


class EdgeEmbedding(nn.Module):
    """
    Module for embedding edges. Edges linking same pairs of atoms share
    the same initial embedding.

    Parameters
    ----------
    dim : int
        Dim of edge embeddings, default to be 128.
    edge_num : int
        Maximum number of edge types, default to be 128.
    pre_train : Edge embeddings or None
        Pre-trained edge embeddings
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

        The edge type is based on the type of the src&dst atom.
        Note that C-O and O-C are the same edge type.
        To map a pair of nodes to one number, we use an unordered pairing function here
        See more detail in this disscussion:
        https://math.stackexchange.com/questions/23503/create-unique-number-from-2-numbers
        Note that, the edge_num should be larger than the square of maximum atomic number
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
        atom_type_x = edges.src["node_type"]
        atom_type_y = edges.dst["node_type"]

        return {
            "type": atom_type_x * atom_type_y +
                    (th.abs(atom_type_x - atom_type_y) - 1)**2 / 4
        }

    def forward(self, g, p_name="edge_f"):
        """Compute edge embeddings

        Parameters
        ----------
        g : DGLGraph
            The graph to compute edge embeddings
        p_name : str

        Returns
        -------
        computed edge embeddings
        """
        g.apply_edges(self.generate_edge_type)
        g.edata[p_name] = self.embedding(g.edata["type"])
        return g.edata[p_name]


class ShiftSoftplus(Softplus):
    """
    Shiftsoft plus activation function:
        1/beta * (log(1 + exp**(beta * x)) - log(shift))

    Parameters
    ----------
    beta : int
    shift : int
    threshold : int
    """
    def __init__(self, beta=1, shift=2, threshold=20):
        super(ShiftSoftplus, self).__init__(beta, threshold)
        self.shift = shift
        self.softplus = Softplus(beta, threshold)

    def forward(self, x):
        """Applies the activation function"""
        return self.softplus(x) - np.log(float(self.shift))


class RBFLayer(nn.Module):
    """
    Radial basis functions Layer.
    e(d) = exp(- gamma * ||d - mu_k||^2)
    default settings:
        gamma = 10
        0 <= mu_k <= 30 for k=1~300
    """
    def __init__(self, low=0, high=30, gap=0.1, dim=1):
        super(RBFLayer, self).__init__()
        self._low = low
        self._high = high
        self._gap = gap
        self._dim = dim

        self._n_centers = int(np.ceil((high - low) / gap))
        centers = np.linspace(low, high, self._n_centers)
        self.centers = th.tensor(centers, dtype=th.float, requires_grad=False)
        self.centers = nn.Parameter(self.centers, requires_grad=False)
        self._fan_out = self._dim * self._n_centers

        self._gap = centers[1] - centers[0]

    def dis2rbf(self, edges):
        """Convert distance matrix to RBF tensor."""
        dist = edges.data["distance"]
        radial = dist - self.centers
        coef = -1 / self._gap
        rbf = th.exp(coef * (radial**2))
        return {"rbf": rbf}

    def forward(self, g):
        """Convert distance scalar to rbf vector"""
        g.apply_edges(self.dis2rbf)
        return g.edata["rbf"]


class CFConv(nn.Module):
    """
    The continuous-filter convolution layer in SchNet.

    Parameters
    ----------
    rbf_dim : int
        Dimension of the RBF layer output
    dim : int
        Dimension of linear layers, default to be 64
    act : str or activation function
        Activation function, default to be shifted softplus
    """
    def __init__(self, rbf_dim, dim=64, act=None):
        super(CFConv, self).__init__()
        self._rbf_dim = rbf_dim
        self._dim = dim

        self.linear_layer1 = nn.Linear(self._rbf_dim, self._dim)
        self.linear_layer2 = nn.Linear(self._dim, self._dim)

        if act is None:
            self.activation = nn.Softplus(beta=0.5, threshold=14)
        else:
            self.activation = act

    def update_edge(self, edges):
        """Update the edge features with two FC layers."""
        rbf = edges.data["rbf"]
        h = self.linear_layer1(rbf)
        h = self.activation(h)
        h = self.linear_layer2(h)
        return {"h": h}

    def forward(self, g):
        """Forward CFConv"""
        g.apply_edges(self.update_edge)
        g.update_all(message_func=fn.u_mul_e('new_node', 'h', 'neighbor_info'),
                     reduce_func=fn.sum('neighbor_info', 'new_node'))
        return g.ndata["new_node"]


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
        self._node_dim = dim
        self.activation = nn.Softplus(beta=0.5, threshold=14)
        self.node_layer1 = nn.Linear(dim, dim, bias=False)
        self.cfconv = CFConv(rbf_dim, dim, act=self.activation)
        self.node_layer2 = nn.Linear(dim, dim)
        self.node_layer3 = nn.Linear(dim, dim)

    def forward(self, g):
        """
        Parameters
        ----------
        g : DGLGraph

        Returns
        -------
        tensor
            Updated atom representations
        """
        g.ndata["new_node"] = self.node_layer1(g.ndata["node"])
        cf_node = self.cfconv(g)
        cf_node_1 = self.node_layer2(cf_node)
        cf_node_1a = self.activation(cf_node_1)
        new_node = self.node_layer3(cf_node_1a)
        g.ndata["node"] = g.ndata["node"] + new_node
        return g.ndata["node"]


class VEConv(nn.Module):
    """
    The Vertex-Edge convolution layer in MGCN which takes edge & vertex features
    in consideration at the same time.

    Parameters
    ----------
    rbf_dim : int
        Dimension of the RBF layer output
    dim : int
        Dimension of intermediate representations, default to be 64.
    update_edge : bool
        Whether to apply a linear layer to update edge representations, default to be True.
    """
    def __init__(self, rbf_dim, dim=64, update_edge=True):
        super(VEConv, self).__init__()
        self._rbf_dim = rbf_dim
        self._dim = dim
        self._update_edge = update_edge

        self.linear_layer1 = nn.Linear(self._rbf_dim, self._dim)
        self.linear_layer2 = nn.Linear(self._dim, self._dim)
        self.linear_layer3 = nn.Linear(self._dim, self._dim)

        self.activation = nn.Softplus(beta=0.5, threshold=14)

    def update_rbf(self, edges):
        """Update the RBF features

        Parameters
        ----------
        edges : EdgeBatch

        Returns
        -------
        dict
            Stores updated features in 'h'
        """
        rbf = edges.data["rbf"]
        h = self.linear_layer1(rbf)
        h = self.activation(h)
        h = self.linear_layer2(h)
        return {"h": h}

    def update_edge(self, edges):
        """Update the edge features.

        Parameters
        ----------
        edges : EdgeBatch

        Returns
        -------
        dict
            Stores updated features in 'edge_f'
        """
        edge_f = edges.data["edge_f"]
        h = self.linear_layer3(edge_f)
        return {"edge_f": h}

    def forward(self, g):
        """VEConv layer forward

        Parameters
        ----------
        g : DGLGraph

        Returns
        -------
        tensor
            Updated atom representations
        """
        g.apply_edges(self.update_rbf)
        if self._update_edge:
            g.apply_edges(self.update_edge)

        g.update_all(message_func=[fn.u_mul_e("new_node", "h", "m_0"),
                                   fn.copy_e("edge_f", "m_1")],
                     reduce_func=[fn.sum("m_0", "new_node_0"),
                                  fn.sum("m_1", "new_node_1")])
        g.ndata["new_node"] = g.ndata.pop("new_node_0") + \
                              g.ndata.pop("new_node_1")

        return g.ndata["new_node"]


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
        self.activation = nn.Softplus(beta=0.5, threshold=14)
        self.node_layer1 = nn.Linear(dim, dim, bias=True)
        self.edge_layer1 = nn.Linear(dim, dim, bias=True)
        self.conv_layer = VEConv(rbf_dim, dim)
        self.node_layer2 = nn.Linear(dim, dim)
        self.node_layer3 = nn.Linear(dim, dim)

    def forward(self, g, level=1):
        """
        Parameters
        ----------
        g : DGLGraph
        level : int
            Level of interaction

        Returns
        -------
        tensor
            Updated atom representations
        """
        g.ndata["new_node"] = self.node_layer1(
            g.ndata["node_%s" % (level - 1)])
        node = self.conv_layer(g)
        g.edata["edge_f"] = self.activation(
            self.edge_layer1(g.edata["edge_f"]))
        node_1 = self.node_layer2(node)
        node_1a = self.activation(node_1)
        new_node = self.node_layer3(node_1a)

        g.ndata["node_%s" % (level)] = g.ndata["node_%s" % (level - 1)] + new_node

        return g.ndata["node_%s" % (level)]
