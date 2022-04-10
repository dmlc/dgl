
"""Shared memory utilities.

For compatibility with older code that uses ``dgl.utils.shared_mem`` namespace; the
content has been moved to ``dgl.ndarray`` module.
"""
from ..ndarray import get_shared_mem_array, create_shared_mem_array # pylint: disable=unused-import
