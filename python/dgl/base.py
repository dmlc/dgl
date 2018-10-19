"""Module for base types and utilities."""
from __future__ import absolute_import

import warnings

from ._ffi.base import DGLError

# A special argument for selecting all nodes/edges.
ALL = "__ALL__"

def is_all(arg):
    return isinstance(arg, str) and arg == ALL

dgl_warning = warnings.warn
