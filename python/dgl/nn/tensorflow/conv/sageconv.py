"""Tensorflow Module for GraphSAGE layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import tensorflow as tf
from tensorflow.keras import layers

from .... import function as fn
from ....utils import expand_as_pair, check_eq_shape


class SAGEConv(layers.Layer):
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
    in_feats : int, or pair of ints
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
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = layers.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = layers.Dense(self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = layers.LSTM(units=self._in_src_feats)
        if aggregator_type != 'gcn':
            self.fc_self = layers.Dense(out_feats, use_bias=bias)
        self.fc_neigh = layers.Dense(out_feats, use_bias=bias)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m']  # (B, L, D)
        rst = self.lstm(m)
        return {'neigh': rst}

    def call(self, graph, feat):
        r"""Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : tf.Tensor or pair of tf.Tensor
            If a single tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of tensors are given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        tf.Tensor
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
            graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
        elif self._aggre_type == 'gcn':
            check_eq_shape(feat)
            graph.srcdata['h'] = feat_src
            graph.dstdata['h'] = feat_dst       # same as above if homogeneous
            graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
            # divide in_degrees
            degs = tf.cast(graph.in_degrees(), tf.float32)
            h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']
                       ) / (tf.expand_dims(degs, -1) + 1)
        elif self._aggre_type == 'pool':
            graph.srcdata['h'] = tf.nn.relu(self.fc_pool(feat_src))
            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
        elif self._aggre_type == 'lstm':
            graph.srcdata['h'] = feat_src
            graph.update_all(fn.copy_src('h', 'm'), self._lstm_reducer)
            h_neigh = graph.dstdata['neigh']
        else:
            raise KeyError(
                'Aggregator type {} not recognized.'.format(self._aggre_type))
        # GraphSAGE GCN does not require fc_self.
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
