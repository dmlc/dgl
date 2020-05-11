"""Readout for SchNet"""
# pylint: disable= no-member, arguments-differ, invalid-name
import dgl
import torch.nn as nn

__all__ = ['MLPNodeReadout']

# pylint: disable=W0221
class MLPNodeReadout(nn.Module):
    """MLP-based Readout.

    This layer updates node representations with a MLP and computes graph representations
    out of node representations with max, mean or sum.

    Parameters
    ----------
    node_feats : int
        Size for the input node features.
    hidden_feats : int
        Size for the hidden representations.
    graph_feats : int
        Size for the output graph representations.
    activation : callable
        Activation function. Default to None.
    mode : 'max' or 'mean' or 'sum'
        Whether to compute elementwise maximum, mean or sum of the node representations.
    """
    def __init__(self, node_feats, hidden_feats, graph_feats, activation=None, mode='sum'):
        super(MLPNodeReadout, self).__init__()

        assert mode in ['max', 'mean', 'sum'], \
            "Expect mode to be 'max' or 'mean' or 'sum', got {}".format(mode)
        self.mode = mode
        self.in_project = nn.Linear(node_feats, hidden_feats)
        self.activation = activation
        self.out_project = nn.Linear(hidden_feats, graph_feats)

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
        node_feats = self.in_project(node_feats)
        if self.activation is not None:
            node_feats = self.activation(node_feats)
        node_feats = self.out_project(node_feats)

        with g.local_scope():
            g.ndata['h'] = node_feats
            if self.mode == 'max':
                graph_feats = dgl.max_nodes(g, 'h')
            elif self.mode == 'mean':
                graph_feats = dgl.mean_nodes(g, 'h')
            elif self.mode == 'sum':
                graph_feats = dgl.sum_nodes(g, 'h')

        return graph_feats
