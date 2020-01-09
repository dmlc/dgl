"""Torch Module for DenseGraphConv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import init


class DenseGraphConv(nn.Module):
    """Graph Convolutional Network layer where the graph structure
    is given by an adjacency matrix.
    We recommend user to use this module when applying graph convolution on
    dense graphs.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    norm : bool
        If True, the normalizer :math:`c_{ij}` is applied. Default: ``True``.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    See also
    --------
    GraphConv
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm=True,
                 bias=True,
                 activation=None):
        super(DenseGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)

        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, adj, feat):
        r"""Compute (Dense) Graph Convolution layer.

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
        if self._norm:
            in_degrees = adj.sum(dim=1)
            norm = th.pow(in_degrees, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp).to(feat.device)
            feat = feat * norm

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat = th.matmul(feat, self.weight)
            rst = adj @ feat
        else:
            # aggregate first then mult W
            rst = adj @ feat
            rst = th.matmul(rst, self.weight)

        if self._norm:
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst
