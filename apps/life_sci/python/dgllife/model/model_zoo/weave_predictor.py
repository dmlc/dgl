"""Weave"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch.nn as nn
import torch.nn.functional as F

from ..gnn import WeaveGNN
from ..readout import WeaveGather

__all__ = ['WeavePredictor']

# pylint: disable=W0221
class WeavePredictor(nn.Module):
    r"""Weave for regression and classification on graphs.

    Weave is introduced in `Molecular Graph Convolutions: Moving Beyond Fingerprints
    <https://arxiv.org/abs/1603.00856>`__

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    num_gnn_layers : int
        Number of GNN (Weave) layers to use. Default to 2.
    gnn_hidden_feats : int
        Size for the hidden node and edge representations. Default to 50.
    gnn_activation : callable
        Activation function to be used in GNN (Weave) layers. Default to ReLU.
    graph_feats : int
        Size for the hidden graph representations. Default to 50.
    gaussian_expand : bool
        Whether to expand each dimension of node features by gaussian histogram in
        computing graph representations. Default to True.
    gaussian_memberships : list of 2-tuples
        For each tuple, the first and second element separately specifies the mean
        and std for constructing a normal distribution. This argument comes into
        effect only when ``gaussian_expand==True``. By default, we set this to be
        ``[(-1.645, 0.283), (-1.080, 0.170), (-0.739, 0.134), (-0.468, 0.118),
        (-0.228, 0.114), (0., 0.114), (0.228, 0.114), (0.468, 0.118),
        (0.739, 0.134), (1.080, 0.170), (1.645, 0.283)]``.
    readout_activation : callable
        Activation function to be used in computing graph representations out of
        node representations. Default to Tanh.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 num_gnn_layers=2,
                 gnn_hidden_feats=50,
                 gnn_activation=F.relu,
                 graph_feats=128,
                 gaussian_expand=True,
                 gaussian_memberships=None,
                 readout_activation=nn.Tanh(),
                 n_tasks=1):
        super(WeavePredictor, self).__init__()

        self.gnn = WeaveGNN(node_in_feats=node_in_feats,
                            edge_in_feats=edge_in_feats,
                            num_layers=num_gnn_layers,
                            hidden_feats=gnn_hidden_feats,
                            activation=gnn_activation)
        self.node_to_graph = nn.Sequential(
            nn.Linear(gnn_hidden_feats, graph_feats),
            readout_activation,
            nn.BatchNorm1d(graph_feats)
        )
        self.readout = WeaveGather(node_in_feats=graph_feats,
                                   gaussian_expand=gaussian_expand,
                                   gaussian_memberships=gaussian_memberships,
                                   activation=readout_activation)
        self.predict = nn.Linear(graph_feats, n_tasks)

    def forward(self, g, node_feats, edge_feats):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges.

        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        node_feats = self.gnn(g, node_feats, edge_feats, node_only=True)
        node_feats = self.node_to_graph(node_feats)
        g_feats = self.readout(g, node_feats)

        return self.predict(g_feats)
