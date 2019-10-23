"""MXNet Module for DenseGraphSAGE"""
# pylint: disable= no-member, arguments-differ, invalid-name
import mxnet as mx
import math
from mxnet import nd
from mxnet.gluon import nn


class DenseSAGEConv(nn.Block):
    """GraphSAGE layer where the graph structure is given by an
    adjacency matrix.
    We recommend to use this module when inducing GraphSAGE operations
    on dense graphs / k-hop graphs.

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
        with self.name_scope():
            self.feat_drop = nn.Dropout(feat_drop)
            self.activation = activation
            self.fc = nn.Dense(out_feats, in_units=in_feats, bias=bias,
                               weight_initializer=mx.init.Xavier(math.sqrt(2.0)))

    def forward(self, adj, feat):
        r"""compute (dense) graph sage layer.

        parameters
        ----------
        adj : mxnet.NDArray
            the adjacency matrix of the graph to apply graph convolution on,
            should be of shape :math:`(n, n)`, where a row represents the destination
            and a column represents the source.
        feat : mxnet.NDArray
            the input feature of shape :math:`(n, d_{in})` where :math:`d_{in}`
            is size of input feature, :math:`n` is the number of nodes.

        returns
        -------
        mxnet.NDArray
            the output feature of shape :math:`(n, d_{out})` where :math:`d_{out}`
            is size of output feature.
        """
        pass