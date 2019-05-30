"""Built-in function base class"""
from __future__ import absolute_import

__all__ = ['BuiltinFunction']


class TargetCode(object):
    """Code for all target"""
    SRC = 0
    DST = 1
    EDGE = 2

    CODE2STR = {
        0: "src",
        1: "dst",
        2: "edge",
    }


class BuiltinFunction(object):
    """Base builtin function class."""
    @property
    def name(self):
        """Return the name of this builtin function."""
        raise NotImplementedError
