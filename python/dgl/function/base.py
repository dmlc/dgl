"""Built-in function base class"""
from __future__ import absolute_import

class BuiltinFunction(object):
    """Base builtin function class."""

    def __call__(self):
        """Regular computation of this builtin function

        This will be used when optimization is not available.
        """
        raise NotImplementedError

    @property
    def name(self):
        """Return the name of this builtin function."""
        raise NotImplementedError

class BundledFunction(object):
    def __init__(self, fn_list):
        self.fn_list = fn_list

    def __call__(self, *args, **kwargs):
        ret = {}
        for fn in self.fn_list:
            ret.update(fn(*args, **kwargs))
        return ret

    @property
    def name(self):
        return "bundled"
