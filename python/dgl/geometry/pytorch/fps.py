"""Utilities for pytorch NN package"""
#pylint: disable=no-member, invalid-name

import torch as th
from torch import nn

from ..capi import farthest_point_sampler

class FarthestPointSampler(nn.Module):
    def __init__(self, npoints):
        super(FarthestPointSampler, self).__init__()
        self.npoints = npoints

    def forward(self, pos):
        device = pos.device
        B, N, C = pos.shape
        pos = pos.reshape(-1, C)
        dist = th.zeros((B * N), dtype=pos.dtype, device=device)
        start_idx = th.randint(0, N - 1, (B, ), dtype=th.int, device=device)
        result = th.zeros((self.npoints * B), dtype=th.int, device=device)
        farthest_point_sampler(pos, B, self.npoints, dist, start_idx, result)
        return result
