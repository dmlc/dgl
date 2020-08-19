"""Farthest Point Sampler for pytorch Geometry package"""
#pylint: disable=no-member, invalid-name

import torch as th
from torch import nn

from ..capi import farthest_point_sampler

class FarthestPointSampler(nn.Module):
    """Farthest Point Sampler without the need to compute all pairs of distance.

    In each batch, the algorithm starts with the sample index specified by ``start_idx``.
    Then for each point, we maintain the minimum to-sample distance.
    Finally, we pick the point with the maximum such distance.
    This process will be repeated for ``sample_points`` - 1 times.

    Parameters
    ----------
    npoints : int
        The number of points to sample in each batch.
    """
    def __init__(self, npoints):
        super(FarthestPointSampler, self).__init__()
        self.npoints = npoints

    def forward(self, pos):
        r"""Memory allocation and sampling

        Parameters
        ----------
        pos : tensor
            The positional tensor of shape (B, N, C)

        Returns
        -------
        tensor of shape (B, self.npoints)
            The sampled indices in each batch.
        """
        device = pos.device
        B, N, C = pos.shape
        pos = pos.reshape(-1, C)
        dist = th.zeros((B * N), dtype=pos.dtype, device=device)
        start_idx = th.randint(0, N - 1, (B, ), dtype=th.long, device=device)
        result = th.zeros((self.npoints * B), dtype=th.long, device=device)
        farthest_point_sampler(pos, B, self.npoints, dist, start_idx, result)
        return result.reshape(B, self.npoints)
