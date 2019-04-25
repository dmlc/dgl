"""Built-in function base class"""
from __future__ import absolute_import
from .. import ndarray as nd

__all__ = ['BuiltinFunction']

class BuiltinFunction(object):
    """Base builtin function class."""
    @property
    def name(self):
        """Return the name of this builtin function."""
        raise NotImplementedError

def _empty_map(ctx):
    return nd.empty([])
