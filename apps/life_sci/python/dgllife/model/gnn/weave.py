"""Weave"""
# pylint: disable= no-member, arguments-differ, invalid-name
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['WeaveGNN']

# pylint: disable=W0221, E1101
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

class WeaveGNN(nn.Module):
    r"""The component of Weave for updating node and edge representations.

    Weave is introduced in `Molecular Graph Convolutions: Moving Beyond Fingerprints
    <https://arxiv.org/abs/1603.00856>`__.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    num_layers : int
        Number of Weave layers to use, which is equivalent to the times of message passing.
        Default to 2.
    hidden_feats : int
        Size for the hidden node and edge representations. Default to 50.
    activation : callable
        Activation function to be used. It cannot be None. Default to ReLU.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 num_layers=2,
                 hidden_feats=50,
                 activation=F.relu):
        super(WeaveGNN, self).__init__()

        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gnn_layers.append(WeaveLayer(node_in_feats=node_in_feats,
                                                  edge_in_feats=edge_in_feats,
                                                  node_node_hidden_feats=hidden_feats,
                                                  edge_node_hidden_feats=hidden_feats,
                                                  node_out_feats=hidden_feats,
                                                  node_edge_hidden_feats=hidden_feats,
                                                  edge_edge_hidden_feats=hidden_feats,
                                                  edge_out_feats=hidden_feats,
                                                  activation=activation))
            else:
                self.gnn_layers.append(WeaveLayer(node_in_feats=hidden_feats,
                                                  edge_in_feats=hidden_feats,
                                                  node_node_hidden_feats=hidden_feats,
                                                  edge_node_hidden_feats=hidden_feats,
                                                  node_out_feats=hidden_feats,
                                                  node_edge_hidden_feats=hidden_feats,
                                                  edge_edge_hidden_feats=hidden_feats,
                                                  edge_out_feats=hidden_feats,
                                                  activation=activation))

    def forward(self, g, node_feats, edge_feats, node_only=True):
        """Updates node representations (and edge representations).

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.
        node_only : bool
            Whether to return updated node representations only or to return both
            node and edge representations. Default to True.

        Returns
        -------
        float32 tensor of shape (V, gnn_hidden_feats)
            Updated node representations.
        float32 tensor of shape (E, gnn_hidden_feats), optional
            This is returned only when ``node_only==False``. Updated edge representations.
        """
        for i in range(len(self.gnn_layers) - 1):
            node_feats, edge_feats = self.gnn_layers[i](g, node_feats, edge_feats)
        return self.gnn_layers[-1](g, node_feats, edge_feats, node_only)
