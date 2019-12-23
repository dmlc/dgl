"""MXNet Module for APPNPConv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

from .... import function as fn

class APPNPConv(nn.Block):
    r"""Approximate Personalized Propagation of Neural Predictions
    layer from paper `Predict then Propagate: Graph Neural Networks
    meet Personalized PageRank <https://arxiv.org/pdf/1810.05997.pdf>`__.

    .. math::
        H^{0} & = X

        H^{t+1} & = (1-\alpha)\left(\hat{D}^{-1/2}
        \hat{A} \hat{D}^{-1/2} H^{t}\right) + \alpha H^{0}

    Parameters
    ----------
    k : int
        Number of iterations :math:`K`.
    alpha : float
        The teleport probability :math:`\alpha`.
    edge_drop : float, optional
        Dropout rate on edges that controls the
        messages received by each node. Default: ``0``.
    """
    def __init__(self,
                 k,
                 alpha,
                 edge_drop=0.):
        super(APPNPConv, self).__init__()
        self._k = k
        self._alpha = alpha
        with self.name_scope():
            self.edge_drop = nn.Dropout(edge_drop)

    def forward(self, graph, feat):
        r"""Compute APPNP layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mx.NDArray
            The input feature of shape :math:`(N, *)` :math:`N` is the
            number of nodes, and :math:`*` could be of any shape.

        Returns
        -------
        mx.NDArray
            The output feature of shape :math:`(N, *)` where :math:`*`
            should be the same as input shape.
        """
        graph = graph.local_var()
        norm = mx.nd.power(mx.nd.clip(
            graph.in_degrees().astype(feat.dtype), a_min=1, a_max=float("inf")), -0.5)
        shp = norm.shape + (1,) * (feat.ndim - 1)
        norm = norm.reshape(shp).as_in_context(feat.context)
        feat_0 = feat
        for _ in range(self._k):
            # normalization by src node
            feat = feat * norm
            graph.ndata['h'] = feat
            graph.edata['w'] = self.edge_drop(
                nd.ones((graph.number_of_edges(), 1), ctx=feat.context))
            graph.update_all(fn.u_mul_e('h', 'w', 'm'),
                             fn.sum('m', 'h'))
            feat = graph.ndata.pop('h')
            # normalization by dst node
            feat = feat * norm
            feat = (1 - self._alpha) * feat + self._alpha * feat_0
        return feat
