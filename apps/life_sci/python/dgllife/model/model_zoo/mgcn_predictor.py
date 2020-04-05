"""MGCN"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch.nn as nn

from ..gnn import MGCNGNN
from ..readout import MLPNodeReadout

__all__ = ['MGCNPredictor']

# pylint: disable=W0221
class MGCNPredictor(nn.Module):
    """MGCN for for regression and classification on graphs.

    MGCN is introduced in `Molecular Property Prediction: A Multilevel Quantum Interactions
    Modeling Perspective <https://arxiv.org/abs/1906.11081>`__.

    Parameters
    ----------
    feats : int
        Size for the node and edge embeddings to learn. Default to 128.
    n_layers : int
        Number of gnn layers to use. Default to 3.
    classifier_hidden_feats : int
        Size for hidden representations in the classifier. Default to 64.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    num_node_types : int
        Number of node types to embed. Default to 100.
    num_edge_types : int
        Number of edge types to embed. Default to 3000.
    cutoff : float
        Largest center in RBF expansion. Default to 5.0
    gap : float
        Difference between two adjacent centers in RBF expansion. Default to 1.0
    """
    def __init__(self, feats=128, n_layers=3, classifier_hidden_feats=64, n_tasks=1,
                 num_node_types=100, num_edge_types=3000, cutoff=5.0, gap=1.0):
        super(MGCNPredictor, self).__init__()

        self.gnn = MGCNGNN(feats=feats,
                           n_layers=n_layers,
                           num_node_types=num_node_types,
                           num_edge_types=num_edge_types,
                           cutoff=cutoff,
                           gap=gap)
        self.readout = MLPNodeReadout(node_feats=(n_layers + 1) * feats,
                                      hidden_feats=classifier_hidden_feats,
                                      graph_feats=n_tasks,
                                      activation=nn.Softplus(beta=1, threshold=20))

    def forward(self, g, node_types, edge_dists):
        """Graph-level regression/soft classification.

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
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        node_feats = self.gnn(g, node_types, edge_dists)
        return self.readout(g, node_feats)
