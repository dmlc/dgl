"""Utilities for pytorch NN package"""
#pylint: disable=no-member, invalid-name

from mxnet import nd
from mxnet.gluon import nn
import numpy as np

from ..capi import farthest_point_sampler

class FarthestPointSampler(nn.Block):
    def __init__(self, npoints):
        super(FarthestPointSampler, self).__init__()
        self.npoints = npoints

    def forward(self, pos):
        ctx = pos.context
        B, N, C = pos.shape
        pos = pos.reshape(-1, C)
        dist = nd.zeros((B * N), dtype=pos.dtype, ctx=ctx)
        start_idx = nd.random.randint(0, N - 1, (B, ), dtype=np.int, ctx=ctx)
        result = nd.zeros((self.npoints * B), dtype=np.int, ctx=ctx)
        farthest_point_sampler(pos, B, self.npoints, dist, start_idx, result)
        return result
