"""MXNet Module for Simplifying Graph Convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name

import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon import nn

from .... import function as fn


class SGConv(nn.Block):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None):
        super(SGConv, self).__init__()
        self._cached = cached
        self._cached_h = None
        self._k = k
        with self.name_scope():
            self.norm = norm
            self.fc = nn.Dense(out_feats, in_units=in_feats, use_bias=bias,
                               weight_initializer=mx.init.Xavier())

    def forward(self, graph, feat):
        r"""Compute Simplifying Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        mxnet.NDArray
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.

        Notes
        -----
        If ``cache`` is se to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
        graph = graph.local_var()
        if self._cached_h is not None:
            feat = self._cached_h
        else:
            # compute normalization
            degs = nd.clip(graph.in_degrees().float(), 1, float('inf'))
            norm = nd.power(degs, -0.5).expand_dims(1)
            norm = norm.as_in_context(feat.context)
            # compute (D^-1 A D)^k X
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

