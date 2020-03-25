"""Torch Module for GMM Conv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import Identity

from .... import function as fn
from ....utils import expand_as_pair


class GMMConv(nn.Block):
    r"""The Gaussian Mixture Model Convolution layer from `Geometric Deep
    Learning on Graphs and Manifolds using Mixture Model CNNs
    <http://openaccess.thecvf.com/content_cvpr_2017/papers/Monti_Geometric_Deep_Learning_CVPR_2017_paper.pdf>`__.

    .. math::
        h_i^{l+1} & = \mathrm{aggregate}\left(\left\{\frac{1}{K}
         \sum_{k}^{K} w_k(u_{ij}), \forall j\in \mathcal{N}(i)\right\}\right)

        w_k(u) & = \exp\left(-\frac{1}{2}(u-\mu_k)^T \Sigma_k^{-1} (u - \mu_k)\right)

    Parameters
    ----------
    in_feats : int, or pair of ints
        Number of input features.

        If the layer is to be applied on a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Number of output features.
    dim : int
        Dimensionality of pseudo-coordinte.
    n_kernels : int
        Number of kernels :math:`K`.
    aggregator_type : str
        Aggregator type (``sum``, ``mean``, ``max``). Default: ``sum``.
    residual : bool
        If True, use residual connection inside this layer. Default: ``False``.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 dim,
                 n_kernels,
                 aggregator_type='sum',
                 residual=False,
                 bias=True):
        super(GMMConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._dim = dim
        self._n_kernels = n_kernels
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        elif aggregator_type == 'max':
            self._reducer = fn.max
        else:
            raise KeyError("Aggregator type {} not recognized.".format(aggregator_type))

        with self.name_scope():
            self.mu = self.params.get('mu',
                                      shape=(n_kernels, dim),
                                      init=mx.init.Normal(0.1))
            self.inv_sigma = self.params.get('inv_sigma',
                                             shape=(n_kernels, dim),
                                             init=mx.init.Constant(1))
            self.fc = nn.Dense(n_kernels * out_feats,
                               in_units=self._in_src_feats,
                               use_bias=False,
                               weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)))
            if residual:
                if self._in_dst_feats != out_feats:
                    self.res_fc = nn.Dense(out_feats, in_units=self._in_dst_feats, use_bias=False)
                else:
                    self.res_fc = Identity()
            else:
                self.res_fc = None

            if bias:
                self.bias = self.params.get('bias',
                                            shape=(out_feats,),
                                            init=mx.init.Zero())
            else:
                self.bias = None

    def forward(self, graph, feat, pseudo):
        """Compute Gaussian Mixture Model Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray
            If a single tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of tensors are given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        pseudo : mxnet.NDArray
            The pseudo coordinate tensor of shape :math:`(E, D_{u})` where
            :math:`E` is the number of edges of the graph and :math:`D_{u}`
            is the dimensionality of pseudo coordinate.

        Returns
        -------
        mxnet.NDArray
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        feat_src, feat_dst = expand_as_pair(feat)
        with graph.local_scope():
            graph.srcdata['h'] = self.fc(feat_src).reshape(
                -1, self._n_kernels, self._out_feats)
            E = graph.number_of_edges()
            # compute gaussian weight
            gaussian = -0.5 * ((pseudo.reshape(E, 1, self._dim) -
                                self.mu.data(feat_src.context)
                                .reshape(1, self._n_kernels, self._dim)) ** 2)
            gaussian = gaussian *\
                       (self.inv_sigma.data(feat_src.context)
                        .reshape(1, self._n_kernels, self._dim) ** 2)
            gaussian = nd.exp(gaussian.sum(axis=-1, keepdims=True)) # (E, K, 1)
            graph.edata['w'] = gaussian
            graph.update_all(fn.u_mul_e('h', 'w', 'm'), self._reducer('m', 'h'))
            rst = graph.dstdata['h'].sum(1)
            # residual connection
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)
            # bias
            if self.bias is not None:
                rst = rst + self.bias.data(feat_dst.context)
            return rst
