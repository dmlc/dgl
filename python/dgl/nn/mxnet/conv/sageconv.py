"""MXNet Module for GraphSAGE layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

from .... import function as fn

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
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        with self.name_scope():
            self.norm = norm
            self.feat_drop = nn.Dropout(feat_drop)
            self.activation = activation
            if aggregator_type == 'pool':
                self.fc_pool = nn.Dense(in_feats, use_bias=bias,
                                        weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)),
                                        in_units=in_feats)
            if aggregator_type == 'lstm':
                raise NotImplementedError
            if aggregator_type != 'gcn':
                self.fc_self = nn.Dense(out_feats, use_bias=bias,
                                        weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)),
                                        in_units=in_feats)
            self.fc_neigh = nn.Dense(out_feats, use_bias=bias,
                                     weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)),
                                     in_units=in_feats)

    def forward(self, graph, feat):
        r"""Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        mxnet.NDArray
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        graph = graph.local_var()
        feat = self.feat_drop(feat)
        h_self = feat
        if self._aggre_type == 'mean':
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
            h_neigh = graph.ndata['neigh']
        elif self._aggre_type == 'gcn':
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
            # divide in degrees
            degs = graph.in_degrees().astype(feat.dtype)
            degs = degs.as_in_context(feat.context)
            h_neigh = (graph.ndata['neigh'] + graph.ndata['h']) / (degs.expand_dims(-1) + 1)
        elif self._aggre_type == 'pool':
            graph.ndata['h'] = nd.relu(self.fc_pool(feat))
            graph.update_all(fn.copy_u('h', 'm'), fn.max('m', 'neigh'))
            h_neigh = graph.ndata['neigh']
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
