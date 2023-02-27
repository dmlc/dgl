"""Init all C APIs in the default namespace."""
from .function import _init_api

__all__ = _init_api("dgl.capi", __name__)
