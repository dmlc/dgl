"""MXNet module for RelGraphConv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math

import mxnet as mx
import numpy as np
from mxnet import gluon, nd
from mxnet.gluon import nn

from .... import function as fn
from .. import utils


class RelGraphConv(gluon.Block):
    r"""Relational graph convolution layer from `Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__

    It can be described as below:

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

    where :math:`B` is the number of bases, :math:`V_b^{(l)}` are linearly combined
    with coefficients :math:`a_{rb}^{(l)}`.

    The block-diagonal-decomposition regularization decomposes :math:`W_r` into :math:`B`
    number of block diagonal matrices. We refer :math:`B` as the number of bases.

    The block regularization decomposes :math:`W_r` by:

    .. math::

       W_r^{(l)} = \oplus_{b=1}^B Q_{rb}^{(l)}

    where :math:`B` is the number of bases, :math:`Q_{rb}^{(l)}` are block
    bases with shape :math:`R^{(d^{(l+1)}/B)*(d^{l}/B)}`.

    Parameters
    ----------
    in_feat : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feat : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    num_rels : int
        Number of relations. .
    regularizer : str
        Which weight regularizer to use "basis" or "bdd".
        "basis" is short for basis-diagonal-decomposition.
        "bdd" is short for block-diagonal-decomposition.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: ``None``.
    bias : bool, optional
        True if bias is added. Default: ``True``.
    activation : callable, optional
        Activation function. Default: ``None``.
    self_loop : bool, optional
        True to include self loop message. Default: ``True``.
    low_mem : bool, optional
        True to use low memory implementation of relation message passing function. Default: False.
        This option trades speed with memory consumption, and will slowdown the forward/backward.
        Turn it on when you encounter OOM problem during training or evaluation. Default: ``False``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    layer_norm: float, optional
        Add layer norm. Default: ``False``

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import mxnet as mx
    >>> from mxnet import gluon
    >>> from dgl.nn import RelGraphConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = mx.nd.ones((6, 10))
    >>> conv = RelGraphConv(10, 2, 3, regularizer='basis', num_bases=2)
    >>> conv.initialize(ctx=mx.cpu(0))
    >>> etype = mx.nd.array(np.array([0,1,2,0,1,2]).astype(np.int64))
    >>> res = conv(g, feat, etype)
    [[ 0.561324    0.33745846]
    [ 0.61585337  0.09992217]
    [ 0.561324    0.33745846]
    [-0.01557937  0.01227859]
    [ 0.61585337  0.09992217]
    [ 0.056508   -0.00307822]]
    <NDArray 6x2 @cpu(0)>
    """

    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        regularizer="basis",
        num_bases=None,
        bias=True,
        activation=None,
        self_loop=True,
        low_mem=False,
        dropout=0.0,
        layer_norm=False,
    ):
        super(RelGraphConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases
        if (
            self.num_bases is None
            or self.num_bases > self.num_rels
            or self.num_bases < 0
        ):
            self.num_bases = self.num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        assert (
            low_mem is False
        ), "MXNet currently does not support low-memory implementation."
        assert (
            layer_norm is False
        ), "MXNet currently does not support layer norm."

        if regularizer == "basis":
            # add basis weights
            self.weight = self.params.get(
                "weight",
                shape=(self.num_bases, self.in_feat, self.out_feat),
                init=mx.init.Xavier(magnitude=math.sqrt(2.0)),
            )
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = self.params.get(
                    "w_comp",
                    shape=(self.num_rels, self.num_bases),
                    init=mx.init.Xavier(magnitude=math.sqrt(2.0)),
                )
            # message func
            self.message_func = self.basis_message_func
        elif regularizer == "bdd":
            if in_feat % num_bases != 0 or out_feat % num_bases != 0:
                raise ValueError(
                    "Feature size must be a multiplier of num_bases."
                )
            # add block diagonal weights
            self.submat_in = in_feat // self.num_bases
            self.submat_out = out_feat // self.num_bases

            # assuming in_feat and out_feat are both divisible by num_bases
            self.weight = self.params.get(
                "weight",
                shape=(
                    self.num_rels,
                    self.num_bases * self.submat_in * self.submat_out,
                ),
                init=mx.init.Xavier(magnitude=math.sqrt(2.0)),
            )
            # message func
            self.message_func = self.bdd_message_func
        else:
            raise ValueError("Regularizer must be either 'basis' or 'bdd'")

        # bias
        if self.bias:
            self.h_bias = self.params.get(
                "bias", shape=(out_feat,), init=mx.init.Zero()
            )

        # weight for self loop
        if self.self_loop:
            self.loop_weight = self.params.get(
                "W_0",
                shape=(in_feat, out_feat),
                init=mx.init.Xavier(magnitude=math.sqrt(2.0)),
            )

        self.dropout = nn.Dropout(dropout)

    def basis_message_func(self, edges):
        """Message function for basis regularizer"""
        ctx = edges.src["h"].context
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.data(ctx).reshape(
                self.num_bases, self.in_feat * self.out_feat
            )
            weight = nd.dot(self.w_comp.data(ctx), weight).reshape(
                self.num_rels, self.in_feat, self.out_feat
            )
        else:
            weight = self.weight.data(ctx)

        msg = utils.bmm_maybe_select(edges.src["h"], weight, edges.data["type"])
        if "norm" in edges.data:
            msg = msg * edges.data["norm"]
        return {"msg": msg}

    def bdd_message_func(self, edges):
        """Message function for block-diagonal-decomposition regularizer"""
        ctx = edges.src["h"].context
        if (
            edges.src["h"].dtype in (np.int32, np.int64)
            and len(edges.src["h"].shape) == 1
        ):
            raise TypeError(
                "Block decomposition does not allow integer ID feature."
            )
        weight = self.weight.data(ctx)[edges.data["type"], :].reshape(
            -1, self.submat_in, self.submat_out
        )
        node = edges.src["h"].reshape(-1, 1, self.submat_in)
        msg = nd.batch_dot(node, weight).reshape(-1, self.out_feat)
        if "norm" in edges.data:
            msg = msg * edges.data["norm"]
        return {"msg": msg}

    def forward(self, g, x, etypes, norm=None):
        """
        Description
        -----------

        Forward computation

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : mx.ndarray.NDArray
            Input node features. Could be either

                * :math:`(|V|, D)` dense tensor
                * :math:`(|V|,)` int64 vector, representing the categorical values of each
                  node. It then treat the input feature as an one-hot encoding feature.
        etypes : mx.ndarray.NDArray
            Edge type tensor. Shape: :math:`(|E|,)`
        norm : mx.ndarray.NDArray
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`.

        Returns
        -------
        mx.ndarray.NDArray
            New node features.
        """
        assert g.is_homogeneous, (
            "not a homogeneous graph; convert it with to_homogeneous "
            "and pass in the edge type as argument"
        )
        with g.local_scope():
            g.ndata["h"] = x
            g.edata["type"] = etypes
            if norm is not None:
                g.edata["norm"] = norm
            if self.self_loop:
                loop_message = utils.matmul_maybe_select(
                    x, self.loop_weight.data(x.context)
                )

            # message passing
            g.update_all(self.message_func, fn.sum(msg="msg", out="h"))

            # apply bias and activation
            node_repr = g.ndata["h"]
            if self.bias:
                node_repr = node_repr + self.h_bias.data(x.context)
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.activation:
                node_repr = self.activation(node_repr)
            node_repr = self.dropout(node_repr)

            return node_repr
