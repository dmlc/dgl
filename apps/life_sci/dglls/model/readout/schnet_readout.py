"""Readout for SchNet"""
import dgl
import torch.nn as nn

from dgl import BatchedDGLGraph
from dgl.nn.pytorch.conv.cfconv import ShiftedSoftplus

__all__ = ['SchNetReadout']

class SchNetReadout(nn.Module):
    """Readout for SchNet.

    SchNet is introduced in `SchNet: A continuous-filter convolutional neural network for
    modeling quantum interactions <https://arxiv.org/abs/1706.08566>`__.

    This class computes graph representations out of node features.

    Parameters
    ----------
    node_feats : int
        Size for the input node features.
    hidden_feats : int
        Size for the hidden representations.
    graph_feats : int
        Size for the output graph representations.
    """
    def __init__(self, node_feats, hidden_feats, graph_feats):
        super(SchNetReadout, self).__init__()

        self.project_nodes = nn.Sequential(
            nn.Linear(node_feats, hidden_feats),
            ShiftedSoftplus(),
            nn.Linear(hidden_feats, graph_feats)
        )

    def forward(self, g, node_feats):
        """Computes graph representations out of node features.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feats)
            Input node features, V for the number of nodes.

        Returns
        -------
        graph_feats : float32 tensor of shape (G, graph_feats)
            Graph representations computed. G for the number of graphs.
        """
        node_feats = self.project_nodes(node_feats)
        g.ndata['h'] = node_feats
        graph_feats = dgl.sum_nodes(g, 'h')
        if not isinstance(g, BatchedDGLGraph):
            graph_feats = graph_feats.unsqueeze(0)
        return graph_feats
