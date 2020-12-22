"""Torch Module for Graph Isomorphism Network layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

from .... import function as fn
from ....utils import expand_as_pair


class GINConv(nn.Module):
    r"""

    Description
    -----------
    Graph Isomorphism Network layer from paper `How Powerful are Graph
    Neural Networks? <https://arxiv.org/pdf/1810.00826.pdf>`__.

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula.
    aggregator_type : str
        Aggregator type to use (``sum``, ``max`` or ``mean``).
    init_eps : float, optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter. Default: ``False``.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GINConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> lin = th.nn.Linear(10, 10)
    >>> conv = GINConv(lin, 'max')
    >>> res = conv(g, feat)
    >>> res
    tensor([[-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.1804,  0.0758, -0.5159,  0.3569, -0.1408, -0.1395, -0.2387,  0.7773,
            0.5266, -0.4465]], grad_fn=<AddmmBackward>)
    """
    def __init__(self,
                 apply_func,
                 aggregator_type,
                 init_eps=0,
                 learn_eps=False):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = th.nn.Parameter(th.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', th.FloatTensor([init_eps]))

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute Graph Isomorphism Network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.update_all(fn.copy_u('h', 'm'), self._reducer('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            return rst
