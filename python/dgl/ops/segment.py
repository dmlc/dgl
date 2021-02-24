"""Segment aggregation operators implemented using DGL graph."""

from ..base import DGLError
from .. import backend as F


def segment_reduce(seglen, value, reducer='sum'):
    """Segment reduction operator.

    It aggregates the value tensor along the first dimension by segments.
    The first argument ``seglen`` stores the length of each segment. Its
    summation must be equal to the first dimension of the ``value`` tensor.
    Zero-length segments are allowed.

    Parameters
    ----------
    seglen : Tensor
        Segment lengths.
    value : Tensor
        Value to aggregate.
    reducer : str, optional
        Aggregation method. Can be 'sum', 'max', 'min', 'mean'.

    Returns
    -------
    Tensor
        Aggregated tensor of shape ``(len(seglen), value.shape[1:])``.

    Examples
    --------

    >>> import dgl
    >>> import torch as th
    >>> val = th.ones(10, 3)
    >>> seg = th.tensor([1, 0, 5, 4])  # 4 segments
    >>> dgl.segment_reduce(seg, val)
    tensor([[1., 1., 1.],
            [0., 0., 0.],
            [5., 5., 5.],
            [4., 4., 4.]])
    """
    offsets = F.cumsum(
        F.cat([F.zeros((1,), F.dtype(seglen), F.context(seglen)), seglen], 0), 0)
    if reducer == 'mean':
        rst = F.segment_reduce('sum', value, offsets)
        rst_shape = F.shape(rst)
        z = F.astype(F.clamp(seglen, 1, len(value)), F.dtype(rst))
        z_shape = (rst_shape[0],) + (1,) * (len(rst_shape) - 1)
        return rst / F.reshape(z, z_shape)
    elif reducer in ['min', 'sum', 'max']:
        rst = F.segment_reduce(reducer, value, offsets)
        if reducer in ['min', 'max']:
            rst = F.replace_inf_with_zero(rst)
        return rst
    else:
        raise DGLError("reducer {} not recognized.".format(reducer))


def segment_softmax(seglen, value):
    """Performa softmax on each segment.

    The first argument ``seglen`` stores the length of each segment. Its
    summation must be equal to the first dimension of the ``value`` tensor.
    Zero-length segments are allowed.

    Parameters
    ----------
    seglen : Tensor
        Segment lengths.
    value : Tensor
        Value to aggregate.

    Returns
    -------
    Tensor
        Result tensor of the same shape as the ``value`` tensor.

    Examples
    --------

    >>> import dgl
    >>> import torch as th
    >>> val = th.ones(10, 3)
    >>> seg = th.tensor([1, 0, 5, 4])  # 4 segments
    >>> dgl.segment_softmax(seg, val)
    tensor([[1.0000, 1.0000, 1.0000],
            [0.2000, 0.2000, 0.2000],
            [0.2000, 0.2000, 0.2000],
            [0.2000, 0.2000, 0.2000],
            [0.2000, 0.2000, 0.2000],
            [0.2000, 0.2000, 0.2000],
            [0.2500, 0.2500, 0.2500],
            [0.2500, 0.2500, 0.2500],
            [0.2500, 0.2500, 0.2500],
            [0.2500, 0.2500, 0.2500]])
    """
    value_max = segment_reduce(seglen, value, reducer='max')
    value = F.exp(value - F.repeat(value_max, seglen, dim=0))
    value_sum = segment_reduce(seglen, value, reducer='sum')
    return value / F.repeat(value_sum, seglen, dim=0)
