"""Farthest Point Sampler for mxnet Geometry package"""
#pylint: disable=no-member, invalid-name

from mxnet import nd
from mxnet.gluon import nn
import numpy as np

from ...base import DGLError
from ..capi import farthest_point_sampler

class FarthestPointSampler(nn.Block):
    """Farthest Point Sampler

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

    def forward(self, pos, start_idx=None):
        r"""Memory allocation and sampling

        Parameters
        ----------
        pos : tensor
            The positional tensor of shape (B, N, C)
        start_idx : int, optional
            If given, appoint the index of the starting point,
            otherwise randomly select a point as the start point.
            (default: None)

        Returns
        -------
        tensor of shape (B, self.npoints)
            The sampled indices in each batch.
        """
        ctx = pos.context
        B, N, C = pos.shape
        pos = pos.reshape(-1, C)
        dist = nd.zeros((B * N), dtype=pos.dtype, ctx=ctx)
        if start_idx is None:
            start_idx = nd.random.randint(0, N - 1, (B, ), dtype=np.int, ctx=ctx)
        else:
            if start_idx >= N or start_idx < 0:
                raise DGLError("Invalid start_idx, expected 0 <= start_idx < {}, got {}".format(
                    N, start_idx))
            start_idx = nd.full((B, ), start_idx, dtype=np.int, ctx=ctx)
        result = nd.zeros((self.npoints * B), dtype=np.int, ctx=ctx)
        farthest_point_sampler(pos, B, self.npoints, dist, start_idx, result)
        return result.reshape(B, self.npoints)
