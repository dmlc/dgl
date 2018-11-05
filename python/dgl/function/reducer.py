"""Built-in reducer function."""
from __future__ import absolute_import

from .. import backend as F
from .base import create_bundled_function_class

__all__ = ["sum", "max"]

class ReduceFunction(object):
    """Base builtin reduce function class."""

    def __call__(self, nodes):
        """Regular computation of this builtin.

        This will be used when optimization is not available.
        """
        raise NotImplementedError

    def name(self):
        """Return the name of this builtin function."""
        raise NotImplementedError

    def is_spmv_supported(self):
        """Return whether the SPMV optimization is supported."""
        raise NotImplementedError


BundledReduceFunction = create_bundled_function_class(
        'BundledReduceFunction', ReduceFunction)


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

    def name(self):
        return self._name

def sum(msg, out):
    """Builtin reduce function that aggregates messages by sum.

    Parameters
    ----------
    msg : str
        The message name.
    out : str
        The output node feature name.
    """
    return SimpleReduceFunction("sum", F.sum, msg, out)

def max(msg, out):
    """Builtin reduce function that aggregates messages by max.

    Parameters
    ----------
    msg : str
        The message name.
    out : str
        The output node feature name.
    """
    return SimpleReduceFunction("max", F.max, msg, out)
