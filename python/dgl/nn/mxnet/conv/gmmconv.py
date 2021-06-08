"""Torch Module for GMM Conv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import Identity

from .... import function as fn
from ....base import DGLError
from ....utils import expand_as_pair


class GMMConv(nn.Block):
    r"""

    Description
    -----------
    The Gaussian Mixture Model Convolution layer from `Geometric Deep
    Learning on Graphs and Manifolds using Mixture Model CNNs
    <http://openaccess.thecvf.com/content_cvpr_2017/papers/Monti_Geometric_Deep_Learning_CVPR_2017_paper.pdf>`__.

    .. math::
        u_{ij} &= f(x_i, x_j), x_j \in \mathcal{N}(i)

        w_k(u) &= \exp\left(-\frac{1}{2}(u-\mu_k)^T \Sigma_k^{-1} (u - \mu_k)\right)

        h_i^{l+1} &= \mathrm{aggregate}\left(\left\{\frac{1}{K}
         \sum_{k}^{K} w_k(u_{ij}), \forall j\in \mathcal{N}(i)\right\}\right)

    where :math:`u` denotes the pseudo-coordinates between a vertex and one of its neighbor,
    computed using function :math:`f`, :math:`\Sigma_k^{-1}` and :math:`\mu_k` are
    learnable parameters representing the covariance matrix and mean vector of a Gaussian kernel.

    Parameters
    ----------
    in_feats : int
        Number of input features; i.e., the number of dimensions of :math:`x_i`.
    out_feats : int
        Number of output features; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    dim : int
        Dimensionality of pseudo-coordinte; i.e, the number of dimensions of :math:`u_{ij}`.
    n_kernels : int
        Number of kernels :math:`K`.
    aggregator_type : str
        Aggregator type (``sum``, ``mean``, ``max``). Default: ``sum``.
    residual : bool
        If True, use residual connection inside this layer. Default: ``False``.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Default: ``False``.

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zero-in-degree when use
    after conv.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import mxnet as mx
    >>> from dgl.nn import GMMConv
    >>>
    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = mx.nd.ones((6, 10))
    >>> conv = GMMConv(10, 2, 3, 2, 'mean')
    >>> conv.initialize(ctx=mx.cpu(0))
    >>> pseudo = mx.nd.ones((12, 3))
    >>> res = conv(g, feat, pseudo)
    >>> res
    [[-0.05083769 -0.1567954 ]
    [-0.05083769 -0.1567954 ]
    [-0.05083769 -0.1567954 ]
    [-0.05083769 -0.1567954 ]
    [-0.05083769 -0.1567954 ]
    [-0.05083769 -0.1567954 ]]
    <NDArray 6x2 @cpu(0)>

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.bipartite((u, v))
    >>> u_fea = mx.nd.random.randn(2, 5)
    >>> v_fea = mx.nd.random.randn(4, 10)
    >>> pseudo = mx.nd.ones((5, 3))
    >>> conv = GMMConv((5, 10), 2, 3, 2, 'mean')
    >>> conv.initialize(ctx=mx.cpu(0))
    >>> res = conv(g, (u_fea, v_fea), pseudo)
    >>> res
    [[-0.1005067  -0.09494358]
    [-0.0023314  -0.07597432]
    [-0.05141905 -0.08545895]
    [-0.1005067  -0.09494358]]
    <NDArray 4x2 @cpu(0)>
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 dim,
                 n_kernels,
                 aggregator_type='sum',
                 residual=False,
                 bias=True,
                 allow_zero_in_degree=False):
        super(GMMConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._dim = dim
        self._n_kernels = n_kernels
        self._allow_zero_in_degree = allow_zero_in_degree
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

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, pseudo):
        """

        Description
        -----------
        Compute Gaussian Mixture Model Convolution layer.

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

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        if not self._allow_zero_in_degree:
            if graph.in_degrees().min() == 0:
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')

        feat_src, feat_dst = expand_as_pair(feat, graph)
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
