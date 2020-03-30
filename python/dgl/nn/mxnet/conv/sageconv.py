"""MXNet Module for GraphSAGE layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

from .... import function as fn
from ....utils import expand_as_pair, check_eq_shape

class SAGEConv(nn.Block):
    r"""GraphSAGE layer from paper `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__.

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} & = \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

        h_{i}^{(l+1)} & = \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1} + b) \right)

        h_{i}^{(l+1)} & = \mathrm{norm}(h_{i}^{l})

    Parameters
    ----------
    in_feats : int
        Input feature size.

        If the layer is to be applied on a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.

        If aggregator type is ``gcn``, the feature size of source and destination nodes
        are required to be the same.
    out_feats : int
        Output feature size.
    feat_drop : float
        Dropout rate on features, default: ``0``.
    aggregator_type : str
        Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type='mean',
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        with self.name_scope():
            self.norm = norm
            self.feat_drop = nn.Dropout(feat_drop)
            self.activation = activation
            if aggregator_type == 'pool':
                self.fc_pool = nn.Dense(self._in_src_feats, use_bias=bias,
                                        weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)),
                                        in_units=self._in_src_feats)
            if aggregator_type == 'lstm':
                raise NotImplementedError
            if aggregator_type != 'gcn':
                self.fc_self = nn.Dense(out_feats, use_bias=bias,
                                        weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)),
                                        in_units=self._in_dst_feats)
            self.fc_neigh = nn.Dense(out_feats, use_bias=bias,
                                     weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)),
                                     in_units=self._in_src_feats)

    def forward(self, graph, feat):
        r"""Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray or pair of mxnet.NDArray
            If a single tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of tensors are given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        mxnet.NDArray
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        graph = graph.local_var()

        if isinstance(feat, tuple):
            feat_src = self.feat_drop(feat[0])
            feat_dst = self.feat_drop(feat[1])
        else:
            feat_src = feat_dst = self.feat_drop(feat)

        h_self = feat_dst

        if self._aggre_type == 'mean':
            graph.srcdata['h'] = feat_src
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
        elif self._aggre_type == 'gcn':
            check_eq_shape(feat)
            graph.srcdata['h'] = feat_src
            graph.dstdata['h'] = feat_dst   # saame as above if homogeneous
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
            # divide in degrees
            degs = graph.in_degrees().astype(feat_dst.dtype)
            degs = degs.as_in_context(feat_dst.context)
            h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.expand_dims(-1) + 1)
        elif self._aggre_type == 'pool':
            graph.srcdata['h'] = nd.relu(self.fc_pool(feat_src))
            graph.update_all(fn.copy_u('h', 'm'), fn.max('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
        elif self._aggre_type == 'lstm':
            raise NotImplementedError
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

        if self._aggre_type == 'gcn':
            rst = self.fc_neigh(h_neigh)
        else:
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)
        return rst
