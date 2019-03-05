"""Module for base types and utilities."""
from __future__ import absolute_import

import warnings

from ._ffi.base import DGLError  # pylint: disable=unused-import
from ._ffi.function import _init_internal_api

# A special symbol for selecting all nodes or edges.
ALL = "__ALL__"

def is_all(arg):
    """Return true if the argument is a special symbol for all nodes or edges."""
    return isinstance(arg, str) and arg == ALL

def dgl_warning(msg):
    """Print out warning messages."""
    warnings.warn(msg)

_init_internal_api()
