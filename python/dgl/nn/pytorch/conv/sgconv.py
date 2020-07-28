"""Torch Module for Simplifying Graph Convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

from .... import transform
from .... import function as fn


class SGConv(nn.Module):
    r"""Simplifying Graph Convolution layer from paper `Simplifying Graph
    Convolutional Networks <https://arxiv.org/pdf/1902.07153.pdf>`__.

    .. math::
        H^{K} = (\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2})^K X \Theta^{l}

    where :math:`\tilde{A}` is :math:`A` + :math:`I`. Thus the graph input is expected to have self-loop edges added.

    Notes
    -----
    Zero in degree nodes could lead to invalid normalizer. A common practice
    to avoid this is to add a self-loop for each node in the graph, which
    can be achieved by:

    >>> g = ... # some DGLGraph
    >>> dgl.add_self_loop(g)

    If we can't do the above in advance for some reason, we need to set add_self_loop to ``True``.

    Parameters
    ----------
    in_feats : int
        Number of input features.
    out_feats : int
        Number of output features.
    k : int
        Number of hops :math:`K`. Defaults:``1``.
    cached : bool
        If True, the module would cache

        .. math::
            (\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}})^K X\Theta

        at the first forward call. This parameter should only be set to
        ``True`` in Transductive Learning setting.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
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
    >>> from dgl.nn import SGConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = SGConv(10, 2, k=2, cached=True)
    >>> res = conv(g, feat)
    >>> res
    tensor([[-0.3002, -0.7406],
            [-0.3002, -0.7406],
            [-0.3002, -0.7406],
            [-0.2123, -0.5237],
            [-0.3002, -0.7406],
            [ 0.0000,  0.0000]], grad_fn=<AddmmBackward>)
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None,
                 add_self_loop=False):
        super(SGConv, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._add_self_loop = add_self_loop
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, graph, feat):
        r"""Compute Simplifying Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.

        Notes
        -----
        If ``cache`` is set to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
        with graph.local_scope():
            if self._add_self_loop:
                graph = transform.add_self_loop(graph)

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    feat = feat * norm
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'),
                                     fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            return self.fc(feat)
