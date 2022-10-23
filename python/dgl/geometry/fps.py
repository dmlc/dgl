"""Farthest Point Sampler for pytorch Geometry package"""
# pylint: disable=no-member, invalid-name

from .. import backend as F
from ..base import DGLError
from .capi import _farthest_point_sampler

__all__ = ["farthest_point_sampler"]


def farthest_point_sampler(pos, npoints, start_idx=None):
    """Farthest Point Sampler without the need to compute all pairs of distance.

    In each batch, the algorithm starts with the sample index specified by ``start_idx``.
    Then for each point, we maintain the minimum to-sample distance.
    Finally, we pick the point with the maximum such distance.
    This process will be repeated for ``sample_points`` - 1 times.

    Parameters
    ----------
    pos : tensor
        The positional tensor of shape (B, N, C)
    npoints : int
        The number of points to sample in each batch.
    start_idx : int, optional
        If given, appoint the index of the starting point,
        otherwise randomly select a point as the start point.
        (default: None)

    Returns
    -------
    tensor of shape (B, npoints)
        The sampled indices in each batch.

    Examples
    --------
    The following exmaple uses PyTorch backend.

    >>> import torch
    >>> from dgl.geometry import farthest_point_sampler
    >>> x = torch.rand((2, 10, 3))
    >>> point_idx = farthest_point_sampler(x, 2)
    >>> print(point_idx)
        tensor([[5, 6],
                [7, 8]])
    """
    ctx = F.context(pos)
    B, N, C = pos.shape
    pos = pos.reshape(-1, C)
    dist = F.zeros((B * N), dtype=pos.dtype, ctx=ctx)
    if start_idx is None:
        start_idx = F.randint(
            shape=(B,), dtype=F.int64, ctx=ctx, low=0, high=N - 1
        )
    else:
        if start_idx >= N or start_idx < 0:
            raise DGLError(
                "Invalid start_idx, expected 0 <= start_idx < {}, got {}".format(
                    N, start_idx
                )
            )
        start_idx = F.full_1d(B, start_idx, dtype=F.int64, ctx=ctx)
    result = F.zeros((npoints * B), dtype=F.int64, ctx=ctx)
    _farthest_point_sampler(pos, B, npoints, dist, start_idx, result)
    return result.reshape(B, npoints)
