"""Torch Module for DenseSAGEConv"""
# pylint: disable= no-member, arguments-differ, invalid-name
from torch import nn


class DenseSAGEConv(nn.Module):
    """GraphSAGE layer where the graph structure is given by an
    adjacency matrix.
    We recommend to use this module when appying GraphSAGE on dense graphs.

    Note that we only support gcn aggregator in DenseSAGEConv.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    feat_drop : float, optional
        Dropout rate on features. Default: 0.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    See also
    --------
    SAGEConv
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(DenseSAGEConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc.weight, gain=gain)

    def forward(self, adj, feat):
        r"""Compute (Dense) Graph SAGE layer.

        Parameters
        ----------
        adj : torch.Tensor
            The adjacency matrix of the graph to apply Graph Convolution on,
            should be of shape :math:`(N, N)`, where a row represents the destination
            and a column represents the source.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        adj = adj.float().to(feat.device)
        feat = self.feat_drop(feat)
        in_degrees = adj.sum(dim=1, keepdim=True)
        h_neigh = (adj @ feat + feat) / (in_degrees + 1)
        rst = self.fc(h_neigh)
        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self._norm is not None:
            rst = self._norm(rst)

        return rst
