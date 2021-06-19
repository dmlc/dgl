"""Torch modules for graph convolutions(GCN)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import init

from .... import function as fn
from ....base import DGLError
from ....utils import expand_as_pair
from ....transform import reverse
from ....convert import block_to_graph
from ....heterograph import DGLBlock

class EdgeWeightNorm(nn.Module):
    r"""

    Description
    -----------
    This module normalizes positive scalar edge weights on a graph
    following the form in `GCN <https://arxiv.org/abs/1609.02907>`__.

    Mathematically, setting ``norm='both'`` yields the following normalization term:

    .. math:
      c_{ji} = (\sqrt{\sum_{k\in\mathcal{N}(j)}e_{jk}}\sqrt{\sum_{k\in\mathcal{N}(i)}e_{ki}})

    And, setting ``norm='right'`` yields the following normalization term:

    .. math:
      c_{ji} = (\sum_{k\in\mathcal{N}(i)}}e_{ki})

    where :math:`e_{ji}` is the scalar weight on the edge from node :math:`j` to node :math:`i`.

    The module returns the normalized weight :math:`e_{ji} / c_{ji}`.

    Parameters
    ----------
    norm : str, optional
        The normalizer as specified above. Default is `'both'`.
    eps : float, optional
        A small offset value in the denominator. Default is 0.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import EdgeWeightNorm, GraphConv

    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> edge_weight = th.tensor([0.5, 0.6, 0.4, 0.7, 0.9, 0.1, 1, 1, 1, 1, 1, 1])
    >>> norm = EdgeWeightNorm(norm='both')
    >>> norm_edge_weight = norm(g, edge_weight)
    >>> conv = GraphConv(10, 2, norm='none', weight=True, bias=True)
    >>> res = conv(g, feat, edge_weight=norm_edge_weight)
    >>> print(res)
    tensor([[-1.1849, -0.7525],
            [-1.3514, -0.8582],
            [-1.2384, -0.7865],
            [-1.9949, -1.2669],
            [-1.3658, -0.8674],
            [-0.8323, -0.5286]], grad_fn=<AddBackward0>)
    """
    def __init__(self, norm='both', eps=0.):
        super(EdgeWeightNorm, self).__init__()
        self._norm = norm
        self._eps = eps

    def forward(self, graph, edge_weight):
        r"""

        Description
        -----------
        Compute normalized edge weight for the GCN model.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        edge_weight : torch.Tensor
            Unnormalized scalar weights on the edges.
            The shape is expected to be :math:`(|E|)`.

        Returns
        -------
        torch.Tensor
            The normalized edge weight.

        Raises
        ------
        DGLError
            Case 1:
            The edge weight is multi-dimensional. Currently this module
            only supports a scalar weight on each edge.

            Case 2:
            The edge weight has non-positive values with ``norm='both'``.
            This will trigger square root and division by a non-positive number.
        """
        with graph.local_scope():
            if isinstance(graph, DGLBlock):
                graph = block_to_graph(graph)
            if len(edge_weight.shape) > 1:
                raise DGLError('Currently the normalization is only defined '
                               'on scalar edge weight. Please customize the '
                               'normalization for your high-dimensional weights.')
            if self._norm == 'both' and th.any(edge_weight <= 0).item():
                raise DGLError('Non-positive edge weight detected with `norm="both"`. '
                               'This leads to square root of zero or negative values.')

            dev = graph.device
            graph.srcdata['_src_out_w'] = th.ones((graph.number_of_src_nodes())).float().to(dev)
            graph.dstdata['_dst_in_w'] = th.ones((graph.number_of_dst_nodes())).float().to(dev)
            graph.edata['_edge_w'] = edge_weight

            if self._norm == 'both':
                reversed_g = reverse(graph)
                reversed_g.edata['_edge_w'] = edge_weight
                reversed_g.update_all(fn.copy_edge('_edge_w', 'm'), fn.sum('m', 'out_weight'))
                degs = reversed_g.dstdata['out_weight'] + self._eps
                norm = th.pow(degs, -0.5)
                graph.srcdata['_src_out_w'] = norm

            if self._norm != 'none':
                graph.update_all(fn.copy_edge('_edge_w', 'm'), fn.sum('m', 'in_weight'))
                degs = graph.dstdata['in_weight'] + self._eps
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                graph.dstdata['_dst_in_w'] = norm

            graph.apply_edges(lambda e: {'_norm_edge_weights': e.src['_src_out_w'] * \
                                                               e.dst['_dst_in_w'] * \
                                                               e.data['_edge_w']})
            return graph.edata['_norm_edge_weights']

# pylint: disable=W0235
class GraphConv(nn.Module):
    r"""

    Description
    -----------
    Graph convolution was introduced in `GCN <https://arxiv.org/abs/1609.02907>`__
    and mathematically is defined as follows:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ji}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the set of neighbors of node :math:`i`,
    :math:`c_{ji}` is the product of the square root of node degrees
    (i.e.,  :math:`c_{ji} = \sqrt{|\mathcal{N}(j)|}\sqrt{|\mathcal{N}(i)|}`),
    and :math:`\sigma` is an activation function.

    If a weight tensor on each edge is provided, the weighted graph convolution is defined as:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{e_{ji}}{c_{ji}}h_j^{(l)}W^{(l)})

    where :math:`e_{ji}` is the scalar weight on the edge from node :math:`j` to node :math:`i`.
    This is NOT equivalent to the weighted graph convolutional network formulation in the paper.

    To customize the normalization term :math:`c_{ji}`, one can first set ``norm='none'`` for
    the model, and send the pre-normalized :math:`e_{ji}` to the forward computation. We provide
    :class:`~dgl.nn.pytorch.EdgeWeightNorm` to normalize scalar edge weight following the GCN paper.

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    norm : str, optional
        How to apply the normalizer. If is `'right'`, divide the aggregated messages
        by each node's in-degrees, which is equivalent to averaging the received messages.
        If is `'none'`, no normalization is applied. Default is `'both'`,
        where the :math:`c_{ji}` in the paper is applied.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Default: ``False``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.

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
    >>> import torch as th
    >>> from dgl.nn import GraphConv

    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> conv = GraphConv(10, 2, norm='both', weight=True, bias=True)
    >>> res = conv(g, feat)
    >>> print(res)
    tensor([[ 1.3326, -0.2797],
            [ 1.4673, -0.3080],
            [ 1.3326, -0.2797],
            [ 1.6871, -0.3541],
            [ 1.7711, -0.3717],
            [ 1.0375, -0.2178]], grad_fn=<AddBackward0>)
    >>> # allow_zero_in_degree example
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> conv = GraphConv(10, 2, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
    >>> res = conv(g, feat)
    >>> print(res)
    tensor([[-0.2473, -0.4631],
            [-0.3497, -0.6549],
            [-0.3497, -0.6549],
            [-0.4221, -0.7905],
            [-0.3497, -0.6549],
            [ 0.0000,  0.0000]], grad_fn=<AddBackward0>)

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.heterograph({('_U', '_E', '_V') : (u, v)})
    >>> u_fea = th.rand(2, 5)
    >>> v_fea = th.rand(4, 5)
    >>> conv = GraphConv(5, 2, norm='both', weight=True, bias=True)
    >>> res = conv(g, (u_fea, v_fea))
    >>> res
    tensor([[-0.2994,  0.6106],
            [-0.4482,  0.5540],
            [-0.5287,  0.8235],
            [-0.2994,  0.6106]], grad_fn=<AddBackward0>)
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GraphConv, self).__init__()
        if norm not in ('none', 'both', 'right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.

        """
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

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

    def forward(self, graph, feat, weight=None, edge_weight=None):
        r"""

        Description
        -----------
        Compute graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, which is the case for bipartite graph, the pair
            must contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
            :math:`(N_{out}, D_{in_{dst}})`.
        weight : torch.Tensor, optional
            Optional external weight tensor.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature

        Raises
        ------
        DGLError
            Case 1:
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

            Case 2:
            External weight is provided while at the same time the module
            has defined its own weight parameter.

        Note
        ----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: :math:`(\text{in_feats}, \text{out_feats})`.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = th.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = th.matmul(rst, weight)

            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)
