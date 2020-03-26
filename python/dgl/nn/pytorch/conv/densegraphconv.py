"""Torch Module for DenseGraphConv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import init
from ....utils import expand_as_pair


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
            The adjacency matrix of the graph to apply Graph Convolution on, when
            applied to a unidirectional bipartite graph, ``adj`` should be of shape
            should be of shape :math:`(N_{out}, N_{in})`; when applied to a homo
            graph, ``adj`` should be of shape :math:`(N, N)`. In both cases,
            a row represents a destination node while a column represents a source
            node.
        feat : torch.Tensor or a pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        feat_src, feat_dst = expand_as_pair(feat)
        adj = adj.float().to(feat_src.device)

        if self._norm:
            src_degrees = adj.sum(dim=0)
            dst_degrees = adj.sum(dim=1)
            norm_src = th.pow(src_degrees, -0.5)
            norm_dst = th.pow(dst_degrees, -0.5)
            shp_src = norm_src.shape + (1,) * (feat_src.dim() - 1)
            shp_dst = norm_dst.shape + (1,) * (feat_dst.dim() - 1)
            norm_src = th.reshape(norm_src, shp_src).to(feat_src.device)
            norm_dst = th.reshape(norm_dst, shp_dst).to(feat_dst.device)
            feat_src = feat_src * norm_src

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat_src = th.matmul(feat_src, self.weight)
            rst = adj @ feat_src
        else:
            # aggregate first then mult W
            rst = adj @ feat_src
            rst = th.matmul(rst, self.weight)

        if self._norm:
            rst = rst * norm_dst

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst
