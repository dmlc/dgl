# pylint: disable=C0103, W0612, E1101
"""Pushing the Boundaries of Molecular Representation for Drug Discovery
with the Graph Attention Mechanism"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import function as fn
from ...contrib.deprecation import deprecated
from ...nn.pytorch.softmax import edge_softmax

class AttentiveGRU1(nn.Module):
    """Update node features with attention and GRU.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node (atom) features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, edge_feats, node_feats):
        """
        Parameters
        ----------
        g : DGLGraph
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        edge_feats : float32 tensor of shape (E, M1)
            Previous edge features.
        node_feats : float32 tensor of shape (V, M2)
            Previous node features.

        Returns
        -------
        float32 tensor of shape (V, M2)
            Updated node features.
        """
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class AttentiveGRU2(nn.Module):
    """Update node features with attention and GRU.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node (atom) features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, node_feats):
        """
        Parameters
        ----------
        g : DGLGraph
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        node_feats : float32 tensor of shape (V, M2)
            Previous node features.

        Returns
        -------
        float32 tensor of shape (V, M2)
            Updated node features.
        """
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class GetContext(nn.Module):
    """Generate context for each node (atom) by message passing at the beginning.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node (atom) features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    graph_feat_size : int
        Size of the learned graph representation (molecular fingerprint).
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def apply_edges1(self, edges):
        """Edge feature update."""
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        """Edge feature update."""
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        """
        Parameters
        ----------
        g : DGLGraph
            Constructed DGLGraphs.
        node_feats : float32 tensor of shape (V, N1)
            Input node features. V for the number of nodes and N1 for the feature size.
        edge_feats : float32 tensor of shape (E, N2)
            Input edge features. E for the number of edges and N2 for the feature size.

        Returns
        -------
        float32 tensor of shape (V, N3)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])

class GNNLayer(nn.Module):
    """GNNLayer for updating node features.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    graph_feat_size : int
        Size for the input graph features.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)

    def apply_edges(self, edges):
        """Edge feature update by concatenating the features of the destination
        and source nodes."""
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        """
        Parameters
        ----------
        g : DGLGraph
            Constructed DGLGraphs.
        node_feats : float32 tensor of shape (V, N1)
            Input node features. V for the number of nodes and N1 for the feature size.

        Returns
        -------
        float32 tensor of shape (V, N1)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        return self.attentive_gru(g, logits, node_feats)

class GlobalPool(nn.Module):
    """Graph feature update.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    graph_feat_size : int
        Size for the input graph features.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GlobalPool, self).__init__()

        self.compute_logits = nn.Sequential(
            nn.Linear(node_feat_size + graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.project_nodes = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, graph_feat_size)
        )
        self.gru = nn.GRUCell(graph_feat_size, graph_feat_size)

    def forward(self, g, node_feats, g_feats, get_node_weight=False):
        """
        Parameters
        ----------
        g : DGLGraph
            Constructed DGLGraphs.
        node_feats : float32 tensor of shape (V, N1)
            Input node features. V for the number of nodes and N1 for the feature size.
        g_feats : float32 tensor of shape (G, N2)
            Input graph features. G for the number of graphs and N2 for the feature size.
        get_node_weight : bool
            Whether to get the weights of atoms during readout.

        Returns
        -------
        float32 tensor of shape (G, N2)
            Updated graph features.
        float32 tensor of shape (V, 1)
            The weights of nodes in readout.
        """
        with g.local_scope():
            g.ndata['z'] = self.compute_logits(
                torch.cat([dgl.broadcast_nodes(g, F.relu(g_feats)), node_feats], dim=1))
            g.ndata['a'] = dgl.softmax_nodes(g, 'z')
            g.ndata['hv'] = self.project_nodes(node_feats)
            context = F.elu(dgl.sum_nodes(g, 'hv', 'a'))

            if get_node_weight:
                return self.gru(context, g_feats), g.ndata['a']
            else:
                return self.gru(context, g_feats)

class AttentiveFP(nn.Module):
    """`Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    Parameters
    ----------
    node_feat_size : int
        Size for the input node (atom) features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    num_layers : int
        Number of GNN layers.
    num_timesteps : int
        Number of timesteps for updating the molecular representation with GRU.
    graph_feat_size : int
        Size of the learned graph representation (molecular fingerprint).
    output_size : int
        Size of the prediction (target labels).
    dropout : float
        The probability for performing dropout.
    """
    @deprecated('Import AttentiveFPPredictor from dgllife.model instead.', 'class')
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers,
                 num_timesteps,
                 graph_feat_size,
                 output_size,
                 dropout):
        super(AttentiveFP, self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

        self.readouts = nn.ModuleList()
        for t in range(num_timesteps):
            self.readouts.append(GlobalPool(graph_feat_size, graph_feat_size, dropout))

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, output_size)
        )

    def forward(self, g, node_feats, edge_feats, get_node_weight=False):
        """
        Parameters
        ----------
        g : DGLGraph
            Constructed DGLGraphs.
        node_feats : float32 tensor of shape (V, N1)
            Input node features. V for the number of nodes and N1 for the feature size.
        edge_feats : float32 tensor of shape (E, N2)
            Input edge features. E for the number of edges and N2 for the feature size.
        get_node_weight : bool
            Whether to get the weights of atoms during readout.

        Returns
        -------
        float32 tensor of shape (G, N3)
            Prediction for the graphs. G for the number of graphs and N3 for the output size.
        node_weights : list of float32 tensors of shape (V, 1)
            Weights of nodes in all readout operations.
        """
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)

        with g.local_scope():
            g.ndata['hv'] = node_feats
            g_feats = dgl.sum_nodes(g, 'hv')

        if get_node_weight:
            node_weights = []

        for readout in self.readouts:
            if get_node_weight:
                g_feats, node_weights_t = readout(g, node_feats, g_feats, get_node_weight)
                node_weights.append(node_weights_t)
            else:
                g_feats = readout(g, node_feats, g_feats)

        if get_node_weight:
            return self.predict(g_feats), node_weights
        else:
            return self.predict(g_feats)
