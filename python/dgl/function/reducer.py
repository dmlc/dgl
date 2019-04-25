"""Built-in reducer function."""
# pylint: disable=redefined-builtin
from __future__ import absolute_import

from .base import BuiltinFunction
from ..runtime import ir
from ..runtime.ir import var

__all__ = ["sum", "max"]


class ReduceFunction(BuiltinFunction):
    """Base builtin reduce function class."""

    def __call__(self)
        """Sumbolic computation of this builtin function to create
        runtime.executor
        """
        raise NotImplementedError

    @property
    def name(self):
        """Return the name of this builtin function."""
        raise NotImplementedError


class SimpleReduceFunction(ReduceFunction):
    """Builtin reduce function that aggregates a single field into another
    single field."""
    def __init__(self, name, msg_field, out_field):
        self._name = name
        self.msg_field = msg_field
        self.out_field = out_field

    def __call__(self, spmat, edge_frame, out_size, edge_map=var.EMPTY_MAP(),
                 out_map=var.EMPTY_MAP()):
        reducer = self._name
        edge_data = ir.READ_COL(edge_frame, var.STR(self.msg_field))
        return ir.COPY_EDGE_REDUCE(reducer, spmat, edge_data, out_size,
                                   edge_map, out_map)

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
    return SimpleReduceFunction("sum", msg, out)


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
    >>>     return {'h': torch.max(nodes.mailbox['m'], dim=1)[0]}
    """
    return SimpleReduceFunction("max", msg, out)
