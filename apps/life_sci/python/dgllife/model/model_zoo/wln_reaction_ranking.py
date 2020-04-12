"""Weisfeiler-Lehman Network (WLN) for ranking candidate products"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch.nn as nn

class WLNReactionRanking(nn.Module):
    r""""""
    def __init__(self):
        super(WLNReactionRanking, self).__init__()

    def forward(self, batch_mol_graphs, node_feats, edge_feats):
        r"""

        Parameters
        ----------
        batch_mol_graphs : DGLGraph
            A batch of molecular graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges.
        """

