import torch.nn as nn
import dgl

class WeightAndSum(nn.Module):
    """Compute importance weights for atoms and perform a weighted sum.

    Parameters
    ----------
    in_feats : int
        Input atom feature size
    """
    def __init__(self, in_feats):
        super(WeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, feats, bg):
        """Compute molecule representations out of atom representations

        Parameters
        ----------
        feats : FloatTensor of shape (N, self.in_feats)
            Representations for all atoms in the molecules
            * N is the total number of atoms in all molecules
        bg : BatchedDGLGraph
            B Batched DGLGraphs for processing multiple molecules in parallel

        Returns
        -------
        FloatTensor of shape (B, self.in_feats)
            Representations for B molecules
        """
        bg = bg.local_var()
        bg.ndata['h'] = feats
        bg.ndata['w'] = self.atom_weighting(bg.ndata['h'])
        h_g_sum = dgl.sum_nodes(bg, 'h', 'w')

        return h_g_sum
