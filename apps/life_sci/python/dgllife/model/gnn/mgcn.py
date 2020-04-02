"""MGCN"""
# pylint: disable= no-member, arguments-differ, invalid-name
import dgl.function as fn
import torch
import torch.nn as nn

from .schnet import RBFExpansion

__all__ = ['MGCNGNN']

# pylint: disable=W0221, E1101
class EdgeEmbedding(nn.Module):
    """Module for embedding edges.

    Edges whose end nodes have the same combination of types
    share the same initial embedding.

    Parameters
    ----------
    num_types : int
        Number of edge types to embed.
    edge_feats : int
        Size for the edge representations to learn.
    """
    def __init__(self, num_types, edge_feats):
        super(EdgeEmbedding, self).__init__()

        self.embed = nn.Embedding(num_types, edge_feats)

    def get_edge_types(self, edges):
        """Generates edge types.

        The edge type is based on the type of the source and destination nodes.
        Note that directions are not distinguished, e.g. C-O and O-C are the same edge type.

        To map each pair of node types to a unique number, we use an unordered pairing function.
        See more details in this discussion:
        https://math.stackexchange.com/questions/23503/create-unique-number-from-2-numbers
        Note that the number of edge types should be larger than the square of the maximum node
        type in the dataset.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges.

        Returns
        -------
        dict
            Mapping 'type' to the computed edge types.
        """
        node_type1 = edges.src['type']
        node_type2 = edges.dst['type']
        return {
            'type': node_type1 * node_type2 + \
                    (torch.abs(node_type1 - node_type2) - 1) ** 2 / 4
        }

    def forward(self, g, node_types):
        """Embeds edge types.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_types : int64 tensor of shape (V)
            Node types to embed, V for the number of nodes.

        Returns
        -------
        float32 tensor of shape (E, edge_feats)
            Edge representations.
        """
        g = g.local_var()
        g.ndata['type'] = node_types
        g.apply_edges(self.get_edge_types)
        return self.embed(g.edata['type'])

class VEConv(nn.Module):
    """Vertex-Edge Convolution in MGCN

    MGCN is introduced in `Molecular Property Prediction: A Multilevel Quantum Interactions
    Modeling Perspective <https://arxiv.org/abs/1906.11081>`__.

    This layer combines both node and edge features in updating node representations.

    Parameters
    ----------
    dist_feats : int
        Size for the expanded distances.
    feats : int
        Size for the input and output node and edge representations.
    update_edge : bool
        Whether to update edge representations. Default to True.
    """
    def __init__(self, dist_feats, feats, update_edge=True):
        super(VEConv, self).__init__()

        self.update_dists = nn.Sequential(
            nn.Linear(dist_feats, feats),
            nn.Softplus(beta=0.5, threshold=14),
            nn.Linear(feats, feats)
        )
        if update_edge:
            self.update_edge_feats = nn.Linear(feats, feats)
        else:
            self.update_edge_feats = None

    def forward(self, g, node_feats, edge_feats, expanded_dists):
        """Performs message passing and updates node and edge representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, feats)
            Input edge features.
        expanded_dists : float32 tensor of shape (E, dist_feats)
            Expanded distances, i.e. the output of RBFExpansion.

        Returns
        -------
        node_feats : float32 tensor of shape (V, feats)
            Updated node representations.
        edge_feats : float32 tensor of shape (E, feats)
            Edge representations, updated if ``update_edge == True`` in initialization.
        """
        expanded_dists = self.update_dists(expanded_dists)
        if self.update_edge_feats is not None:
            edge_feats = self.update_edge_feats(edge_feats)

        g = g.local_var()
        g.ndata.update({'hv': node_feats})
        g.edata.update({'dist': expanded_dists, 'he': edge_feats})
        g.update_all(message_func=[fn.u_mul_e('hv', 'dist', 'm_0'),
                                   fn.copy_e('he', 'm_1')],
                     reduce_func=[fn.sum('m_0', 'hv_0'),
                                  fn.sum('m_1', 'hv_1')])
        node_feats = g.ndata.pop('hv_0') + g.ndata.pop('hv_1')

        return node_feats, edge_feats

class MultiLevelInteraction(nn.Module):
    """Building block for MGCN.

    MGCN is introduced in `Molecular Property Prediction: A Multilevel Quantum Interactions
    Modeling Perspective <https://arxiv.org/abs/1906.11081>`__. This layer combines node features,
    edge features and expanded distances in message passing and updates node and edge
    representations.

    Parameters
    ----------
    feats : int
        Size for the input and output node and edge representations.
    dist_feats : int
        Size for the expanded distances.
    """
    def __init__(self, feats, dist_feats):
        super(MultiLevelInteraction, self).__init__()

        self.project_in_node_feats = nn.Linear(feats, feats)
        self.conv = VEConv(dist_feats, feats)
        self.project_out_node_feats = nn.Sequential(
            nn.Linear(feats, feats),
            nn.Softplus(beta=0.5, threshold=14),
            nn.Linear(feats, feats)
        )
        self.project_edge_feats = nn.Sequential(
            nn.Linear(feats, feats),
            nn.Softplus(beta=0.5, threshold=14)
        )

    def forward(self, g, node_feats, edge_feats, expanded_dists):
        """Performs message passing and updates node and edge representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, feats)
            Input edge features
        expanded_dists : float32 tensor of shape (E, dist_feats)
            Expanded distances, i.e. the output of RBFExpansion.

        Returns
        -------
        node_feats : float32 tensor of shape (V, feats)
            Updated node representations.
        edge_feats : float32 tensor of shape (E, feats)
            Updated edge representations.
        """
        new_node_feats = self.project_in_node_feats(node_feats)
        new_node_feats, edge_feats = self.conv(g, new_node_feats, edge_feats, expanded_dists)
        new_node_feats = self.project_out_node_feats(new_node_feats)
        node_feats = node_feats + new_node_feats

        edge_feats = self.project_edge_feats(edge_feats)

        return node_feats, edge_feats

class MGCNGNN(nn.Module):
    """MGCN.

    MGCN is introduced in `Molecular Property Prediction: A Multilevel Quantum Interactions
    Modeling Perspective <https://arxiv.org/abs/1906.11081>`__.

    This class performs message passing in MGCN and returns the updated node representations.

    Parameters
    ----------
    feats : int
        Size for the node and edge embeddings to learn. Default to 128.
    n_layers : int
        Number of gnn layers to use. Default to 3.
    num_node_types : int
        Number of node types to embed. Default to 100.
    num_edge_types : int
        Number of edge types to embed. Default to 3000.
    cutoff : float
        Largest center in RBF expansion. Default to 30.
    gap : float
        Difference between two adjacent centers in RBF expansion. Default to 0.1.
    """
    def __init__(self, feats=128, n_layers=3, num_node_types=100,
                 num_edge_types=3000, cutoff=30., gap=0.1):
        super(MGCNGNN, self).__init__()

        self.node_embed = nn.Embedding(num_node_types, feats)
        self.edge_embed = EdgeEmbedding(num_edge_types, feats)
        self.rbf = RBFExpansion(high=cutoff, gap=gap)

        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(MultiLevelInteraction(feats, len(self.rbf.centers)))

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
        float32 tensor of shape (V, feats * (n_layers + 1))
            Output node representations.
        """
        node_feats = self.node_embed(node_types)
        edge_feats = self.edge_embed(g, node_types)
        expanded_dists = self.rbf(edge_dists)

        all_layer_node_feats = [node_feats]
        for gnn in self.gnn_layers:
            node_feats, edge_feats = gnn(g, node_feats, edge_feats, expanded_dists)
            all_layer_node_feats.append(node_feats)
        return torch.cat(all_layer_node_feats, dim=1)
