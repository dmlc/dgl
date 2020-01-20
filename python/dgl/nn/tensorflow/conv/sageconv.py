"""Tensorflow Module for GraphSAGE layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import tensorflow as tf
from tensorflow.keras import layers

from .... import function as fn


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
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = layers.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = layers.Dense(in_feats)
        if aggregator_type == 'lstm':
            self.lstm = layers.LSTM(units=in_feats)
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
        feat : tf.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        tf.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        graph = graph.local_var()
        feat = self.feat_drop(feat)
        h_self = feat
        if self._aggre_type == 'mean':
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
            h_neigh = graph.ndata['neigh']
        elif self._aggre_type == 'gcn':
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
            # divide in_degrees
            degs = tf.cast(graph.in_degrees(), tf.float32)
            h_neigh = (graph.ndata['neigh'] + graph.ndata['h']
                       ) / (tf.expand_dims(degs, -1) + 1)
        elif self._aggre_type == 'pool':
            graph.ndata['h'] = tf.nn.relu(self.fc_pool(feat))
            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
            h_neigh = graph.ndata['neigh']
        elif self._aggre_type == 'lstm':
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src('h', 'm'), self._lstm_reducer)
            h_neigh = graph.ndata['neigh']
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
