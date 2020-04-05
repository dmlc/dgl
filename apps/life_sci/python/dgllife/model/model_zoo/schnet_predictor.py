"""SchNet"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch.nn as nn

from dgl.nn.pytorch.conv.cfconv import ShiftedSoftplus

from ..gnn import SchNetGNN
from ..readout import MLPNodeReadout

__all__ = ['SchNetPredictor']

# pylint: disable=W0221
class SchNetPredictor(nn.Module):
    """SchNet for regression and classification on graphs.

    SchNet is introduced in `SchNet: A continuous-filter convolutional neural network for
    modeling quantum interactions <https://arxiv.org/abs/1706.08566>`__.

    Parameters
    ----------
    node_feats : int
        Size for node representations to learn. Default to 64.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of hidden representations for the i-th interaction
        (gnn) layer. ``len(hidden_feats)`` equals the number of interaction (gnn) layers.
        Default to ``[64, 64, 64]``.
    classifier_hidden_feats : int
        Size for hidden representations in the classifier. Default to 64.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    num_node_types : int
        Number of node types to embed. Default to 100.
    cutoff : float
        Largest center in RBF expansion. Default to 30.
    gap : float
        Difference between two adjacent centers in RBF expansion. Default to 0.1.
    """
    def __init__(self, node_feats=64, hidden_feats=None, classifier_hidden_feats=64, n_tasks=1,
                 num_node_types=100, cutoff=30., gap=0.1):
        super(SchNetPredictor, self).__init__()

        self.gnn = SchNetGNN(node_feats, hidden_feats, num_node_types, cutoff, gap)
        self.readout = MLPNodeReadout(node_feats, classifier_hidden_feats, n_tasks,
                                      activation=ShiftedSoftplus())

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
