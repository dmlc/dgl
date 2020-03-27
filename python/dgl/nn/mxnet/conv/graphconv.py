"""MXNet modules for graph convolutions(GCN)"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math

import mxnet as mx
from mxnet import gluon

from .... import function as fn
from ....base import DGLError

class GraphConv(gluon.Block):
    r"""Apply graph convolution over an input signal.

    Graph convolution is introduced in `GCN <https://arxiv.org/abs/1609.02907>`__
    and can be described as below:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the neighbor set of node :math:`i`. :math:`c_{ij}` is equal
    to the product of the square root of node degrees:
    :math:`\sqrt{|\mathcal{N}(i)|}\sqrt{|\mathcal{N}(j)|}`. :math:`\sigma` is an activation
    function.

    The model parameters are initialized as in the
    `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__ where
    the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
    and the bias is initialized to be zero.

    Notes
    -----
    Zero in degree nodes could lead to invalid normalizer. A common practice
    to avoid this is to add a self-loop for each node in the graph, which
    can be achieved by:

    >>> g = ... # some DGLGraph
    >>> g.add_edges(g.nodes(), g.nodes())


    Parameters
    ----------
    in_feats : int
        Number of input features.
    out_feats : int
        Number of output features.
    norm : str, optional
        How to apply the normalizer. If is `'right'`, divide the aggregated messages
        by each node's in-degrees, which is equivalent to averaging the received messages.
        If is `'none'`, no normalization is applied. Default is `'both'`,
        where the :math:`c_{ij}` in the paper is applied.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    weight : mxnet.gluon.parameter.Parameter
        The learnable weight tensor.
    bias : mxnet.gluon.parameter.Parameter
        The learnable bias tensor.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None):
        super(GraphConv, self).__init__()
        if norm not in ('none', 'both', 'right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        with self.name_scope():
            if weight:
                self.weight = self.params.get('weight', shape=(in_feats, out_feats),
                                              init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
            else:
                self.weight = None

            if bias:
                self.bias = self.params.get('bias', shape=(out_feats,),
                                            init=mx.init.Zero())
            else:
                self.bias = None

        self._activation = activation

    def forward(self, graph, feat, weight=None):
        r"""Compute graph convolution.

        Notes
        -----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: "math:`(\text{in_feats}, \text{out_feats})`.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray
            The input feature.
        weight : torch.Tensor, optional
            Optional external weight tensor.

        Returns
        -------
        mxnet.NDArray
            The output feature
        """
        graph = graph.local_var()

        if self._norm == 'both':
            degs = graph.out_degrees().as_in_context(feat.context).astype('float32')
            degs = mx.nd.clip(degs, a_min=1, a_max=float("inf"))
            norm = mx.nd.power(degs, -0.5)
            shp = norm.shape + (1,) * (feat.ndim - 1)
            norm = norm.reshape(shp)
            feat = feat * norm

        if weight is not None:
            if self.weight is not None:
                raise DGLError('External weight is provided while at the same time the'
                               ' module has defined its own weight parameter. Please'
                               ' create the module with flag weight=False.')
        else:
            weight = self.weight.data(feat.context)

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            if weight is not None:
                feat = mx.nd.dot(feat, weight)
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata.pop('h')
        else:
            # aggregate first then mult W
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata.pop('h')
            if weight is not None:
                rst = mx.nd.dot(rst, weight)

        if self._norm != 'none':
            degs = graph.in_degrees().as_in_context(feat.context).astype('float32')
            degs = mx.nd.clip(degs, a_min=1, a_max=float("inf"))
            if self._norm == 'both':
                norm = mx.nd.power(degs, -0.5)
            else:
                norm = 1.0 / degs
            shp = norm.shape + (1,) * (feat.ndim - 1)
            norm = norm.reshape(shp)
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias.data(rst.context)

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def __repr__(self):
        summary = 'GraphConv('
        summary += 'in={:d}, out={:d}, normalization={}, activation={}'.format(
            self._in_feats, self._out_feats,
            self._norm, self._activation)
        summary += ')'
        return summary
