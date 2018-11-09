"""Built-in function base class"""
from __future__ import absolute_import

class BuiltinFunction(object):
    """Base builtin function class."""

    def __call__(self):
        """Regular computation of this builtin function

        This will be used when optimization is not available.
        """
        raise NotImplementedError

    def name(self):
        """Return the name of this builtin function."""
        raise NotImplementedError
