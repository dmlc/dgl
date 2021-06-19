"""Tensorflow Module for GraphSAGE layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import tensorflow as tf
from tensorflow.keras import layers

from .... import function as fn
from ....utils import expand_as_pair, check_eq_shape


class SAGEConv(layers.Layer):
    r"""

    Description
    -----------
    GraphSAGE layer from paper `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__.

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} &= \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

        h_{i}^{(l+1)} &= \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1}) \right)

        h_{i}^{(l+1)} &= \mathrm{norm}(h_{i}^{l})

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.

        GATConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer applies on a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.

        If aggregator type is ``gcn``, the feature size of source and destination nodes
        are required to be the same.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
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

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import tensorflow as tf
    >>> from dgl.nn import SAGEConv
    >>>
    >>> # Case 1: Homogeneous graph
    >>> with tf.device("CPU:0"):
    >>>     g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>>     g = dgl.add_self_loop(g)
    >>>     feat = tf.ones((6, 10))
    >>>     conv = SAGEConv(10, 2, 'pool')
    >>>     res = conv(g, feat)
    >>>     res
    <tf.Tensor: shape=(6, 2), dtype=float32, numpy=
    array([[-3.6633523 , -0.90711546],
        [-3.6633523 , -0.90711546],
        [-3.6633523 , -0.90711546],
        [-3.6633523 , -0.90711546],
        [-3.6633523 , -0.90711546],
        [-3.6633523 , -0.90711546]], dtype=float32)>

    >>> # Case 2: Unidirectional bipartite graph
    >>> with tf.device("CPU:0"):
    >>>     u = [0, 1, 0, 0, 1]
    >>>     v = [0, 1, 2, 3, 2]
    >>>     g = dgl.bipartite((u, v))
    >>>     u_fea = tf.convert_to_tensor(np.random.rand(2, 5))
    >>>     v_fea = tf.convert_to_tensor(np.random.rand(4, 5))
    >>>     conv = SAGEConv((5, 10), 2, 'mean')
    >>>     res = conv(g, (u_fea, v_fea))
    >>>     res
    <tf.Tensor: shape=(4, 2), dtype=float32, numpy=
    array([[-0.59453356, -0.4055441 ],
        [-0.47459763, -0.717764  ],
        [ 0.3221837 , -0.29876417],
        [-0.63356155,  0.09390211]], dtype=float32)>
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
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : tf.Tensor or pair of tf.Tensor
            If a tf.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of tf.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        tf.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = tf.cast(tf.zeros(
                    (graph.number_of_dst_nodes(), self._in_src_feats)), tf.float32)

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
