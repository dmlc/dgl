"""Segment aggregation operators implemented using DGL graph."""

from ..base import DGLError
from .. import backend as F
from .. import convert
from .. import function as fn


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
    ctx = F.context(seglen)
    # TODO(minjie): a more efficient implementation is to create a graph
    #   directly from a CSR structure.
    u = F.copy_to(F.arange(0, F.shape(value)[0], F.int32), ctx)
    v = F.repeat(F.copy_to(F.arange(0, len(seglen), F.int32), ctx),
                 seglen, dim=0)
    if len(u) != len(v):
        raise DGLError("Invalid seglen array:", seglen,
                       ". Its summation must be equal to value.shape[0].")
    g = convert.heterograph({('_U', '_E', '_V'): (u, v)})
    g.srcdata['h'] = value
    g.update_all(fn.copy_u('h', 'm'), getattr(fn, reducer)('m', 'h'))
    return g.dstdata['h']


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
    reducer : str, optional
        Aggregation method. Can be 'sum', 'max', 'min', 'mean'.

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
