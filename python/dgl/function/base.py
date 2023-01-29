"""Built-in function base class"""
from __future__ import absolute_import

__all__ = ["BuiltinFunction", "TargetCode"]


class TargetCode(object):
    """Code for target

    Note: must be consistent with the target code definition in C++ side:
          src/kernel/binary_reduce_common.h
    """

    SRC = 0
    DST = 1
    EDGE = 2

    CODE2STR = {
        0: "u",
        1: "v",
        2: "e",
    }


class BuiltinFunction(object):
    """Base builtin function class."""

    @property
    def name(self):
        """Return the name of this builtin function."""
        raise NotImplementedError
