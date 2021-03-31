"""Torch Module for APPNPConv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

from .... import function as fn

class APPNPConv(nn.Module):
    r"""

    Description
    -----------
    Approximate Personalized Propagation of Neural Predictions
    layer from paper `Predict then Propagate: Graph Neural Networks
    meet Personalized PageRank <https://arxiv.org/pdf/1810.05997.pdf>`__.

    .. math::
        H^{0} &= X

        H^{l+1} &= (1-\alpha)\left(\tilde{D}^{-1/2}
        \tilde{A} \tilde{D}^{-1/2} H^{l}\right) + \alpha H^{0}

    where :math:`\tilde{A}` is :math:`A` + :math:`I`.

    Parameters
    ----------
    k : int
        The number of iterations :math:`K`.
    alpha : float
        The teleport probability :math:`\alpha`.
    edge_drop : float, optional
        The dropout rate on edges that controls the
        messages received by each node. Default: ``0``.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import APPNPConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = APPNPConv(k=3, alpha=0.5)
    >>> res = conv(g, feat)
    >>> res
    tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000],
            [1.0303, 1.0303, 1.0303, 1.0303, 1.0303, 1.0303, 1.0303, 1.0303, 1.0303,
            1.0303],
            [0.8643, 0.8643, 0.8643, 0.8643, 0.8643, 0.8643, 0.8643, 0.8643, 0.8643,
            0.8643],
            [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
            0.5000]])
    """
    def __init__(self,
                 k,
                 alpha,
                 edge_drop=0.):
        super(APPNPConv, self).__init__()
        self._k = k
        self._alpha = alpha
        self.edge_drop = nn.Dropout(edge_drop)

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute APPNP layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, *)`. :math:`N` is the
            number of nodes, and :math:`*` could be of any shape.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *)` where :math:`*`
            should be the same as input shape.
        """
        with graph.local_scope():
            src_norm = th.pow(graph.out_degrees().float().clamp(min=1), -0.5)
            shp = src_norm.shape + (1,) * (feat.dim() - 1)
            src_norm = th.reshape(src_norm, shp).to(feat.device)
            dst_norm = th.pow(graph.in_degrees().float().clamp(min=1), -0.5)
            shp = dst_norm.shape + (1,) * (feat.dim() - 1)
            dst_norm = th.reshape(dst_norm, shp).to(feat.device)
            feat_0 = feat
            for _ in range(self._k):
                # normalization by src node
                feat = feat * src_norm
                graph.ndata['h'] = feat
                graph.edata['w'] = self.edge_drop(
                    th.ones(graph.number_of_edges(), 1).to(feat.device))
                graph.update_all(fn.u_mul_e('h', 'w', 'm'),
                                 fn.sum('m', 'h'))
                feat = graph.ndata.pop('h')
                # normalization by dst node
                feat = feat * dst_norm
                feat = (1 - self._alpha) * feat + self._alpha * feat_0
            return feat
