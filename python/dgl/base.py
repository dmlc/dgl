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

_INTERNAL_COLUMNS = {NTYPE, NID, ETYPE, EID}

def is_internal_column(name):
    """Return true if the column name is reversed by DGL."""
    return name in _INTERNAL_COLUMNS

def is_all(arg):
    """Return true if the argument is a special symbol for all nodes or edges."""
    return isinstance(arg, str) and arg == ALL

# pylint: disable=invalid-name
_default_formatwarning = warnings.formatwarning

class DGLWarning(UserWarning):
    """DGL Warning class."""

# pylint: disable=unused-argument
def dgl_warning_format(message, category, filename, lineno, line=None):
    """Format DGL warnings."""
    if isinstance(category, DGLWarning):
        return "DGL Warning: {}\n".format(message)
    else:
        return _default_formatwarning(message, category, filename, lineno, line=None)

def dgl_warning(message, category=DGLWarning, stacklevel=1):
    """DGL warning wrapper that defaults to ``DGLWarning`` instead of ``UserWarning`` category."""
    return warnings.warn(message, category=category, stacklevel=1)

warnings.formatwarning = dgl_warning_format

_init_internal_api()
