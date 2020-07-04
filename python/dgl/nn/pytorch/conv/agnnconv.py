"""Torch Module for Attention-based Graph Neural Network layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import functional as F

from .... import transform
from .... import function as fn
from ....ops import edge_softmax
from ....utils import expand_as_pair


class AGNNConv(nn.Module):
    r"""Attention-based Graph Neural Network layer from paper `Attention-based
    Graph Neural Network for Semi-Supervised Learning
    <https://arxiv.org/abs/1803.03735>`__.

    .. math::
        H^{l+1} = P H^{l}

    where :math:`P` is computed as:

    .. math::
        P_{ij} = \mathrm{softmax}_i ( \beta \cdot \cos(h_i^l, h_j^l))

    Notes
    -----
    Zero in-degree nodes could lead to invalid normalizer. A common practice
    to avoid this is to add a self-loop for each node in the graph if it is homogeneous,
    which can be achieved by:

    >>> g = ... # some DGLGraph
    >>> dgl.add_self_loop(g)

    If we can't do the above in advance for some reason, we need to set add_self_loop to ``True``.

    For heterogeneous graph, it doesn't make sense to add self-loop. Then we need to filter out the destination nodes with zero in-degree when use in downstream.

    Parameters
    ----------
    init_beta : float, optional
        The :math:`\beta` in the formula.
    learn_beta : bool, optional
        If True, :math:`\beta` will be learnable parameter.
    add_self_loop: bool, optional
        Add self-loop to graph when compute Conv. If no self-loop is added, the feature for a node with zero
        in-degree will be all zero after Conv. This is harmful for some applications. We recommend adding
        self_loop in graph construction phase to reduce duplicated operations. If we can't do that, we
        need to set add_self_loop to ``True`` here.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import AGNNConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = AGNNConv()
    >>> res = conv(g, feat)
    >>> res
    tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        grad_fn=<BinaryReduceBackward>)
    """
    def __init__(self,
                 init_beta=1.,
                 learn_beta=True,
                 add_self_loop=False):
        super(AGNNConv, self).__init__()
        self._add_self_loop = add_self_loop
        if learn_beta:
            self.beta = nn.Parameter(th.Tensor([init_beta]))
        else:
            self.register_buffer('beta', th.Tensor([init_beta]))

    def forward(self, graph, feat):
        r"""Compute AGNN layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, *)` :math:`N` is the
            number of nodes, and :math:`*` could be of any shape.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *)` and :math:`(N_{out}, *)`, the the :math:`*` in the later
            tensor must equal the previous one.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *)` where :math:`*`
            should be the same as input shape.
        """
        with graph.local_scope():
            if self._add_self_loop:
                graph = transform.add_self_loop(graph)


            feat_src, feat_dst = expand_as_pair(feat, graph)


            graph.srcdata['h'] = feat_src
            graph.srcdata['norm_h'] = F.normalize(feat_src, p=2, dim=-1)
            if isinstance(feat, tuple) or graph.is_block:
                graph.dstdata['norm_h'] = F.normalize(feat_dst, p=2, dim=-1)
            # compute cosine distance
            graph.apply_edges(fn.u_dot_v('norm_h', 'norm_h', 'cos'))
            cos = graph.edata.pop('cos')
            e = self.beta * cos
            graph.edata['p'] = edge_softmax(graph, e)
            graph.update_all(fn.u_mul_e('h', 'p', 'm'), fn.sum('m', 'h'))
            return graph.dstdata.pop('h')
