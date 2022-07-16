"""Torch Module for DenseSAGEConv"""
# pylint: disable= no-member, arguments-differ, invalid-name
from torch import nn

from ....utils import check_eq_shape


class DenseSAGEConv(nn.Module):
    """GraphSAGE layer from `Inductive Representation Learning on Large Graphs
    <https://arxiv.org/abs/1706.02216>`__

    We recommend to use this module when appying GraphSAGE on dense graphs.

    Note that we only support gcn aggregator in DenseSAGEConv.

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    feat_drop : float, optional
        Dropout rate on features. Default: 0.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import DenseSAGEConv
    >>>
    >>> feat = th.ones(6, 10)
    >>> adj = th.tensor([[0., 0., 1., 0., 0., 0.],
    ...         [1., 0., 0., 0., 0., 0.],
    ...         [0., 1., 0., 0., 0., 0.],
    ...         [0., 0., 1., 0., 0., 1.],
    ...         [0., 0., 0., 1., 0., 0.],
    ...         [0., 0., 0., 0., 0., 0.]])
    >>> conv = DenseSAGEConv(10, 2)
    >>> res = conv(adj, feat)
    >>> res
    tensor([[1.0401, 2.1008],
            [1.0401, 2.1008],
            [1.0401, 2.1008],
            [1.0401, 2.1008],
            [1.0401, 2.1008],
            [1.0401, 2.1008]], grad_fn=<AddmmBackward>)

    See also
    --------
    `SAGEConv <https://docs.dgl.ai/api/python/nn.pytorch.html#sageconv>`__
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        feat_drop=0.0,
        bias=True,
        norm=None,
        activation=None,
    ):
        super(DenseSAGEConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Notes
        -----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        """
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc.weight, gain=gain)

    def forward(self, adj, feat):
        r"""

        Description
        -----------
        Compute (Dense) Graph SAGE layer.

        Parameters
        ----------
        adj : torch.Tensor
            The adjacency matrix of the graph to apply SAGE Convolution on, when
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
        check_eq_shape(feat)
        if isinstance(feat, tuple):
            feat_src = self.feat_drop(feat[0])
            feat_dst = self.feat_drop(feat[1])
        else:
            feat_src = feat_dst = self.feat_drop(feat)
        adj = adj.to(feat_src)
        in_degrees = adj.sum(dim=1, keepdim=True)
        h_neigh = (adj @ feat_src + feat_dst) / (in_degrees + 1)
        rst = self.fc(h_neigh)
        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self._norm is not None:
            rst = self._norm(rst)

        return rst
