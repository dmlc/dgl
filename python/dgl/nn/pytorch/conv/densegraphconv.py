"""Torch Module for DenseGraphConv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import init


class DenseGraphConv(nn.Module):
    """

    Description
    -----------
    Graph Convolutional Network layer where the graph structure
    is given by an adjacency matrix.
    We recommend user to use this module when applying graph convolution on
    dense graphs.

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    norm : str, optional
        How to apply the normalizer. If is `'right'`, divide the aggregated messages
        by each node's in-degrees, which is equivalent to averaging the received messages.
        If is `'none'`, no normalization is applied. Default is `'both'`,
        where the :math:`c_{ij}` in the paper is applied.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Notes
    -----
    Zero in-degree nodes will lead to all-zero output. A common practice
    to avoid this is to add a self-loop for each node in the graph,
    which can be achieved by setting the diagonal of the adjacency matrix to be 1.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import DenseGraphConv
    >>>
    >>> feat = th.ones(6, 10)
    >>> adj = th.tensor([[0., 0., 1., 0., 0., 0.],
    ...         [1., 0., 0., 0., 0., 0.],
    ...         [0., 1., 0., 0., 0., 0.],
    ...         [0., 0., 1., 0., 0., 1.],
    ...         [0., 0., 0., 1., 0., 0.],
    ...         [0., 0., 0., 0., 0., 0.]])
    >>> conv = DenseGraphConv(10, 2)
    >>> res = conv(adj, feat)
    >>> res
    tensor([[0.2159, 1.9027],
            [0.3053, 2.6908],
            [0.3053, 2.6908],
            [0.3685, 3.2481],
            [0.3053, 2.6908],
            [0.0000, 0.0000]], grad_fn=<AddBackward0>)

    See also
    --------
    `GraphConv <https://docs.dgl.ai/api/python/nn.pytorch.html#graphconv>`__
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
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
        r"""

        Description
        -----------
        Compute (Dense) Graph Convolution layer.

        Parameters
        ----------
        adj : torch.Tensor
            The adjacency matrix of the graph to apply Graph Convolution on, when
            applied to a unidirectional bipartite graph, ``adj`` should be of shape
            should be of shape :math:`(N_{out}, N_{in})`; when applied to a homo
            graph, ``adj`` should be of shape :math:`(N, N)`. In both cases,
            a row represents a destination node while a column represents a source
            node.
        feat : torch.Tensor
            The input feature.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        adj = adj.float().to(feat.device)
        src_degrees = adj.sum(dim=0).clamp(min=1)
        dst_degrees = adj.sum(dim=1).clamp(min=1)
        feat_src = feat

        if self._norm == 'both':
            norm_src = th.pow(src_degrees, -0.5)
            shp = norm_src.shape + (1,) * (feat.dim() - 1)
            norm_src = th.reshape(norm_src, shp).to(feat.device)
            feat_src = feat_src * norm_src

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat_src = th.matmul(feat_src, self.weight)
            rst = adj @ feat_src
        else:
            # aggregate first then mult W
            rst = adj @ feat_src
            rst = th.matmul(rst, self.weight)

        if self._norm != 'none':
            if self._norm == 'both':
                norm_dst = th.pow(dst_degrees, -0.5)
            else: # right
                norm_dst = 1.0 / dst_degrees
            shp = norm_dst.shape + (1,) * (feat.dim() - 1)
            norm_dst = th.reshape(norm_dst, shp).to(feat.device)
            rst = rst * norm_dst

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst
