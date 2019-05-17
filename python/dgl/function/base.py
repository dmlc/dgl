"""Built-in function base class"""
from __future__ import absolute_import

__all__ = ['BuiltinFunction']


class BuiltinFunction(object):
    """Base builtin function class."""
    @property
    def name(self):
        """Return the name of this builtin function."""
        raise NotImplementedError
