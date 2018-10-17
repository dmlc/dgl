"""Module for base types and utilities."""
from __future__ import absolute_import

from ._ffi.base import DGLError

# A special argument for selecting all nodes/edges.
ALL = "__ALL__"

def is_all(arg):
    return isinstance(arg, str) and arg == ALL

__MSG__ = "__MSG__"
__REPR__ = "__REPR__"
