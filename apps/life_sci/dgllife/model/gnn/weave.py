"""Weave"""
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import DGLGraph
from torch.distributions import Normal

__all__ = ['WeaveLayer', 'WeaveGather', 'WeaveModel' ]


class WeaveLayer(nn.Module):
    r"""Single Weave layer from `Molecular Graph Convolutions: Moving Beyond Fingerprints
    <https://arxiv.org/abs/1603.00856>`__

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    node_node_hidden_feats : int
        Size for the hidden node representations in updating node representations.
        Default to 50.
    edge_node_hidden_feats : int
        Size for the hidden edge representations in updating node representations.
        Default to 50.
    node_out_feats : int
        Size for the output node representations. Default to 50.
    node_edge_hidden_feats : int
        Size for the hidden node representations in updating edge representations.
        Default to 50.
    edge_edge_hidden_feats : int
        Size for the hidden edge representations in updating edge representations.
        Default to 50.
    edge_out_feats : int
        Size for the output edge representations. Default to 50.
    activation : callable
        Activation function to apply. Default to ReLU.
    """

    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_node_hidden_feats=50,
                 edge_node_hidden_feats=50,
                 node_out_feats=50,
                 node_edge_hidden_feats=50,
                 edge_edge_hidden_feats=50,
                 edge_out_feats=50,
                 activation=F.relu):
        super(WeaveLayer, self).__init__()

        self.activation = activation

        # Layers for updating node representations
        self.node_to_node = nn.Linear(node_in_feats, node_node_hidden_feats)
        self.edge_to_node = nn.Linear(edge_in_feats, edge_node_hidden_feats)
        self.update_node = nn.Linear(
            node_node_hidden_feats + edge_node_hidden_feats, node_out_feats)

        # Layers for updating edge representations
        self.left_node_to_edge = nn.Linear(node_in_feats, node_edge_hidden_feats)
        self.right_node_to_edge = nn.Linear(node_in_feats, node_edge_hidden_feats)
        self.edge_to_edge = nn.Linear(edge_in_feats, edge_edge_hidden_feats)
        self.update_edge = nn.Linear(
            2 * node_edge_hidden_feats + edge_edge_hidden_feats, edge_out_feats)

    def forward(self, g, node_feats, edge_feats, node_only=False):
        r"""Update node and edge representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.
        node_only : bool
            Whether to update node representations only. If False, edge representations
            will be updated as well. Default to False.

        Returns
        -------
        new_node_feats : float32 tensor of shape (V, node_out_feats)
            Updated node representations.
        new_edge_feats : float32 tensor of shape (E, edge_out_feats)
            Updated edge representations.
        """
        g = g.local_var()

        # Update node features
        node_node_feats = self.activation(self.node_to_node(node_feats))
        g.edata['e2n'] = self.activation(self.edge_to_node(edge_feats))
        g.update_all(fn.copy_edge('e2n', 'm'), fn.sum('m', 'e2n'))
        edge_node_feats = g.ndata.pop('e2n')
        new_node_feats = self.activation(self.update_node(
            torch.cat([node_node_feats, edge_node_feats], dim=1)))

        if node_only:
            return new_node_feats

        # Update edge features
        g.ndata['left_hv'] = self.left_node_to_edge(node_feats)
        g.ndata['right_hv'] = self.right_node_to_edge(node_feats)
        g.apply_edges(fn.u_add_v('left_hv', 'right_hv', 'first'))
        g.apply_edges(fn.u_add_v('right_hv', 'left_hv', 'second'))
        first_edge_feats = self.activation(g.edata.pop('first'))
        second_edge_feats = self.activation(g.edata.pop('second'))
        third_edge_feats = self.activation(self.edge_to_edge(edge_feats))
        new_edge_feats = self.activation(self.update_edge(
            torch.cat([first_edge_feats, second_edge_feats, third_edge_feats], dim=1)))

        return new_node_feats, new_edge_feats


class WeaveGather(nn.Module):
    r"""represent features of molecules by corresponding atom features calculated through weave layers.

    Parameters
    ----------
    n_input: int
        number of features for input molecules
    gaussian_expand: bool
        Whether to expand each dimension of atomic features by gaussian histogram
    activation : callable
        Activation function to apply. Default to tanh.
    """

    def __init__(self,
                 n_input=128,
                 gaussian_expand=False,
                 activation=torch.tanh):
        super(WeaveGather, self).__init__()
        self.n_input = n_input
        self.activation = activation
        self.gaussian_expand = gaussian_expand
        self.sum = nn.Linear(n_input * 11, n_input)
        self.activation = activation

    def forward(self, g, inputs_atom):
        """
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        inputs_atom : float32 tensor of shape (V, n_inputs)
            Input atom features. V for the number of atoms in the batch of graphs.

        Returns
        -------
        output_molecules : float32 tensor of shape (N,1)
            Output molecule features. N for the number of atom features in batched graphs.
        """
        g = g.local_var()

        # gather atom features to corresponding molecules
        if self.gaussian_expand:
            inputs_atom = self.gaussian_histogram(inputs_atom)
            output_molecules = self.activation(self.sum(inputs_atom))
        else:
            output_molecules = torch.sum(inputs_atom)
        g.ndata['h'] = output_molecules
        return output_molecules

    def gaussian_histogram(self, x):
        gaussian_memberships = [(-1.645, 0.283), (-1.080, 0.170), (-0.739, 0.134),
                                (-0.468, 0.118), (-0.228, 0.114), (0., 0.114),
                                (0.228, 0.114), (0.468, 0.118), (0.739, 0.134),
                                (1.080, 0.170), (1.645, 0.283)]
        dist = [Normal(p[0], p[1]) for p in gaussian_memberships]
        dist_max = [dist[i].log_prob(gaussian_memberships[i][0]).exp() for i in range(11)]
        outputs = [dist[i].log_prob(x).exp() / dist_max[i] for i in range(11)]
        outputs = torch.stack(list(outputs), dim=2)
        outputs = outputs / torch.sum(outputs, dim=2, keepdim=True)
        outputs = torch.reshape(outputs, [-1, self.n_input * 11])
        return outputs


class WeaveModel(nn.Module):
    r"""applying weave model, updating and representing features of molecules.
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    node_out_feats : int
        Size for the output node representations. Default to 50.
    n_graph_feat: int
        Final atom layer convolution depth.
    eps: int
        parameter for nn.BatchNorm2d
    momentum: int
        parameter for nn.BatchNorm2d
    affine: bool
        parameter for nn.BatchNorm2d
    gaussian_expand: bool
        Whether to expand each dimension of atomic features by gaussian histogram
    activation : callable
        Activation function to apply. Default to tanh.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=50,
                 n_graph_feat=128,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 gaussian_expand=True,
                 activation=torch.tanh):
        super(WeaveModel, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.gaussian_expand = gaussian_expand
        self.weave_layers= WeaveLayer(node_in_feats, edge_in_feats)
        # tensorflow.keras.layers.Dense in deepchem: dense to n_graph_feat
        self.atom_to_densed_atom = nn.Linear(node_out_feats, n_graph_feat)
        self.activation = activation
        # tensorflow.keras.layers,BatchNormalization in deepchem: normalizing atom features in batched graphs
        self.batch_norm = nn.BatchNorm2d(n_graph_feat, eps=1e-05, momentum=0.1, affine=True)
        # gather features of atoms and representing as molecule features
        self.weave_gather = WeaveGather(n_input=128, gaussian_expand=True, activation=torch.tanh)

    def forward(self, g, node_feats, edge_feats, n_layers):
        """
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        n_layers: int
            number of weave layer used in model.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.

        Returns
        -------
        output_molecules : float32 tensor of shape (N,1)
            Output molecule features. N for the number of atom features in batched graphs.
        """

        g = g.local_var()
        # weave layers
        for i in range(n_layers):
            if i == n_layers - 1:
                node_feats = self.weave_layers(g, node_feats, edge_feats, node_only=True)
            else:
                node_feats, edge_feats = self.weave_layers(g, node_feats, edge_feats, node_only=False)
        # dense
        node_feats = self.activation(self.atom_to_densed_atom(node_feats))
        # BatchNorm
        node_feats = self.batch_norm(node_feats)
        # gather
        molecule_features = self.weave_gather(node_feats)
        g.ndata['h'] = molecule_features
        return molecule_features

