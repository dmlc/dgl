"""Utility functions related to pinned memory tensors."""

from ..base import DGLError
from .. import backend as F
from .._ffi.function import _init_api

def pin_memory_inplace(tensor):
    """Register the tensor into pinned memory in-place (i.e. without copying)."""
    if F.backend_name in ['mxnet', 'tensorflow']:
        raise DGLError("The {} backend does not support pinning " \
            "tensors in-place.".format(F.backend_name))

    # needs to be writable to allow in-place modification
    try:
        F.zerocopy_to_dgl_ndarray_for_write(tensor).pin_memory_()
    except Exception as e:
        raise DGLError("Failed to pin memory in-place due to: {}".format(e))

def unpin_memory_inplace(tensor):
    """Unregister the tensor from pinned memory in-place (i.e. without copying)."""
    # needs to be writable to allow in-place modification
    try:
        F.zerocopy_to_dgl_ndarray_for_write(tensor).unpin_memory_()
    except Exception as e:
        raise DGLError("Failed to unpin memory in-place due to: {}".format(e))

def gather_pinned_tensor_rows(tensor, rows):
    """Directly gather rows from a CPU tensor given an indices array on CUDA devices,
    and returns the result on the same CUDA device without copying.

    Parameters
    ----------
    tensor : Tensor
        The tensor.  Must be in pinned memory.
    rows : Tensor
        The rows to gather.  Must be a CUDA tensor.

    Returns
    -------
    Tensor
        The result with the same device as :attr:`rows`.
    """
    return F.from_dgl_nd(_CAPI_DGLIndexSelectCPUFromGPU(F.to_dgl_nd(tensor), F.to_dgl_nd(rows)))

_init_api("dgl.ndarray.uvm", __name__)
