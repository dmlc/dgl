"""Module for common feature initializers."""
from __future__ import absolute_import

from . import backend as F

__all__ = ['base_initializer', 'zero_initializer']

def base_initializer(shape, dtype, ctx, range):
    """The function signature for feature initializer.

    Parameters
    ----------
    shape : tuple of int
        The shape of the result features. The first dimension
        is the batch dimension.
    dtype : data type object
        The data type of the returned features.
    ctx : context object
        The device context of the returned features.
    range : slice
        The start id and the end id of the features to be initialized.
        The id could be node or edge id depending on the scenario.
        Note that the step is always None.
    """
    raise NotImplementedError

def zero_initializer(shape, dtype, ctx, range):
    """Initialize zero-value features."""
    return F.zeros(shape, dtype, ctx)
