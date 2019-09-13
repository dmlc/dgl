"""Module for base types and utilities."""
from __future__ import absolute_import

import warnings

from ._ffi.base import DGLError  # pylint: disable=unused-import
from ._ffi.function import _init_internal_api

# A special symbol for selecting all nodes or edges.
ALL = "__ALL__"
# An alias for [:]
SLICE_FULL = slice(None, None, None)
# Reserved column names for storing parent node/edge types and IDs in flattened heterographs
NTYPE = '_TYPE'
NID = '_ID'
ETYPE = '_TYPE'
EID = '_ID'

def is_all(arg):
    """Return true if the argument is a special symbol for all nodes or edges."""
    return isinstance(arg, str) and arg == ALL

def dgl_warning(msg, warn_type=UserWarning):
    """Print out warning messages."""
    warnings.warn(msg, warn_type)

_init_internal_api()
