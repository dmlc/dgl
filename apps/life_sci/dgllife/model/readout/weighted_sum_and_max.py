"""Apply weighted sum and max pooling to the node representations and concatenate the results."""
import dgl
import torch
import torch.nn as nn

from dgl import BatchedDGLGraph
from dgl.nn.pytorch import WeightAndSum

__all__ = ['WeightedSumAndMax']

class WeightedSumAndMax(nn.Module):
    r"""Apply weighted sum and max pooling to the node
    representations and concatenate the results.

    Parameters
    ----------
    in_feats : int
        Input node feature size
    """
    def __init__(self, in_feats):
        super(WeightedSumAndMax, self).__init__()

        self.weight_and_sum = WeightAndSum(in_feats)

    def forward(self, bg, feats):
        """Readout

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization

        Returns
        -------
        h_g : FloatTensor of shape (B, 2 * M1)
            * B is the number of graphs in the batch
            * M1 is the input node feature size, which must match
              in_feats in initialization
        """
        h_g_sum = self.weight_and_sum(bg, feats)
        with bg.local_scope():
            bg.ndata['h'] = feats
            h_g_max = dgl.max_nodes(bg, 'h')

        if not isinstance(bg, BatchedDGLGraph):
            h_g_sum = h_g_sum.unsqueeze(0)
            h_g_max = h_g_max.unsqueeze(0)
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)

        return h_g
