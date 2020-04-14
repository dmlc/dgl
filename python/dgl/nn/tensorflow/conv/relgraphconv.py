"""Tensorflow Module for Relational graph convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import tensorflow as tf
from tensorflow.keras import layers

from .... import function as fn
from .. import utils


class RelGraphConv(layers.Layer):
    r"""Relational graph convolution layer.

    Relational graph convolution is introduced in "`Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__"
    and can be described as below:

    .. math::

       h_i^{(l+1)} = \sigma(\sum_{r\in\mathcal{R}}
       \sum_{j\in\mathcal{N}^r(i)}\frac{1}{c_{i,r}}W_r^{(l)}h_j^{(l)}+W_0^{(l)}h_i^{(l)})

    where :math:`\mathcal{N}^r(i)` is the neighbor set of node :math:`i` w.r.t. relation
    :math:`r`. :math:`c_{i,r}` is the normalizer equal
    to :math:`|\mathcal{N}^r(i)|`. :math:`\sigma` is an activation function. :math:`W_0`
    is the self-loop weight.

    The basis regularization decomposes :math:`W_r` by:

    .. math::

       W_r^{(l)} = \sum_{b=1}^B a_{rb}^{(l)}V_b^{(l)}

    where :math:`B` is the number of bases.

    The block-diagonal-decomposition regularization decomposes :math:`W_r` into :math:`B`
    number of block diagonal matrices. We refer :math:`B` as the number of bases.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    num_rels : int
        Number of relations.
    regularizer : str
        Which weight regularizer to use "basis" or "bdd"
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 regularizer="basis",
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases
        if self.num_bases is None or self.num_bases > self.num_rels or self.num_bases < 0:
            self.num_bases = self.num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        xinit = tf.keras.initializers.glorot_uniform()
        zeroinit = tf.keras.initializers.zeros()

        if regularizer == "basis":
            # add basis weights
            self.weight = tf.Variable(initial_value=xinit(
                shape=(self.num_bases, self.in_feat, self.out_feat),
                dtype='float32'), trainable=True)
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = tf.Variable(initial_value=xinit(
                    shape=(self.num_rels, self.num_bases), dtype='float32'), trainable=True)
            # message func
            self.message_func = self.basis_message_func
        elif regularizer == "bdd":
            if in_feat % num_bases != 0 or out_feat % num_bases != 0:
                raise ValueError(
                    'Feature size must be a multiplier of num_bases.')
            # add block diagonal weights
            self.submat_in = in_feat // self.num_bases
            self.submat_out = out_feat // self.num_bases

            # assuming in_feat and out_feat are both divisible by num_bases
            self.weight = tf.Variable(initial_value=xinit(
                shape=(self.num_rels, self.num_bases *
                       self.submat_in * self.submat_out),
                dtype='float32'), trainable=True)
            # message func
            self.message_func = self.bdd_message_func
        else:
            raise ValueError("Regularizer must be either 'basis' or 'bdd'")

        # bias
        if self.bias:
            self.h_bias = tf.Variable(initial_value=zeroinit(
                shape=(out_feat), dtype='float32'), trainable=True)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = tf.Variable(initial_value=xinit(
                shape=(in_feat, out_feat), dtype='float32'), trainable=True)

        self.dropout = layers.Dropout(rate=dropout)

    def basis_message_func(self, edges):
        """Message function for basis regularizer"""
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = tf.reshape(self.weight, (self.num_bases,
                                              self.in_feat * self.out_feat))
            weight = tf.reshape(tf.matmul(self.w_comp, weight), (
                self.num_rels, self.in_feat, self.out_feat))
        else:
            weight = self.weight

        msg = utils.bmm_maybe_select(
            edges.src['h'], weight, edges.data['type'])
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def bdd_message_func(self, edges):
        """Message function for block-diagonal-decomposition regularizer"""
        if ((edges.src['h'].dtype == tf.int64) and
                len(edges.src['h'].shape) == 1):
            raise TypeError(
                'Block decomposition does not allow integer ID feature.')
        weight = tf.reshape(tf.gather(
            self.weight, edges.data['type']), (-1, self.submat_in, self.submat_out))
        node = tf.reshape(edges.src['h'], (-1, 1, self.submat_in))
        msg = tf.reshape(tf.matmul(node, weight), (-1, self.out_feat))
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def call(self, g, x, etypes, norm=None):
        """ Forward computation

        Parameters
        ----------
        g : DGLGraph
            The graph.
        x : tf.Tensor
            Input node features. Could be either
                * :math:`(|V|, D)` dense tensor
                * :math:`(|V|,)` int64 vector, representing the categorical values of each
                  node. We then treat the input feature as an one-hot encoding feature.
        etypes : tf.Tensor
            Edge type tensor. Shape: :math:`(|E|,)`
        norm : tf.Tensor
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`

        Returns
        -------
        tf.Tensor
            New node features.
        """
        assert g.is_homograph(), \
            "not a homograph; convert it with to_homo and pass in the edge type as argument"
        with g.local_scope():
            g.ndata['h'] = x
            g.edata['type'] = tf.cast(etypes, tf.int64)
            if norm is not None:
                g.edata['norm'] = norm
            if self.self_loop:
                loop_message = utils.matmul_maybe_select(x, self.loop_weight)
            # message passing
            g.update_all(self.message_func, fn.sum(msg='msg', out='h'))
            # apply bias and activation
            node_repr = g.ndata['h']
            if self.bias:
                node_repr = node_repr + self.h_bias
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.activation:
                node_repr = self.activation(node_repr)
            node_repr = self.dropout(node_repr)
            return node_repr
