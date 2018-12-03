"""Built-in reducer function."""
from __future__ import absolute_import

from .. import backend as F
from .base import BuiltinFunction

__all__ = ["sum", "max"]

class ReduceFunction(BuiltinFunction):
    """Base builtin reduce function class."""

    def __call__(self, nodes):
        """Regular computation of this builtin.

        This will be used when optimization is not available.
        """
        raise NotImplementedError

    @property
    def name(self):
        """Return the name of this builtin function."""
        raise NotImplementedError

    def is_spmv_supported(self):
        """Return whether the SPMV optimization is supported."""
        raise NotImplementedError


class SimpleReduceFunction(ReduceFunction):
    """Builtin reduce function that aggregates a single field into another
    single field."""
    def __init__(self, name, op, msg_field, out_field):
        self._name = name
        self.op = op
        self.msg_field = msg_field
        self.out_field = out_field

    def is_spmv_supported(self):
        # NOTE: only sum is supported right now.
        return self._name == "sum"

    def __call__(self, nodes):
        return {self.out_field : self.op(nodes.mailbox[self.msg_field], 1)}

    @property
    def name(self):
        return self._name

def sum(msg, out):
    """Builtin reduce function that aggregates messages by sum.

    Parameters
    ----------
    msg : str
        The message field.
    out : str
        The output node feature field.
    Examples
    --------
    >>> import dgl
    >>> reduce_func = dgl.function.sum(msg='m', out='h')

    The above example is equivalent to the following user defined function
    (if using PyTorch):

    >>> import torch
    >>> def reduce_func(nodes):
    >>>     return {'h': torch.sum(nodes.mailbox['m'], dim=1)}
    """
    return SimpleReduceFunction("sum", F.sum, msg, out)

def max(msg, out):
    """Builtin reduce function that aggregates messages by max.

    Parameters
    ----------
    msg : str
        The message field.
    out : str
        The output node feature field.

    Examples
    --------
    >>> import dgl
    >>> reduce_func = dgl.function.max(msg='m', out='h')

    The above example is equivalent to the following user defined function
    (if using PyTorch):

    >>> import torch
    >>> def reduce_func(nodes):
    >>>     return {'h': torch.max(nodes.mailbox['m'], dim=1)}
    """
    return SimpleReduceFunction("max", F.max, msg, out)
