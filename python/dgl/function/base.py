"""Built-in function base class"""
from __future__ import absolute_import

__all__ = ['BuiltinFunction', 'BundledFunction']

class BuiltinFunction(object):
    """Base builtin function class."""
    @property
    def name(self):
        """Return the name of this builtin function."""
        raise NotImplementedError

class BundledFunction(object):
    """A utility class that bundles multiple functions.

    Parameters
    ----------
    fn_list : list of callable
        The function list.
    """
    def __init__(self, fn_list):
        self.fn_list = fn_list

    def __call__(self, *args, **kwargs):
        """Regular computation of this builtin function

        This will be used when optimization is not available and should
        ONLY be called by DGL framework.
        """
        ret = {}
        for fn in self.fn_list:
            ret.update(fn(*args, **kwargs))
        return ret

    @property
    def name(self):
        """Return the name."""
        return "bundled"
