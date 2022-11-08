# pylint: disable=invalid-name, unused-import
"""Runtime stream APIs which are mainly for internal test use only.
For applications, please use PyTorch's stream management, of which DGL is aware.
"""
from __future__ import absolute_import

import ctypes

from .base import _FFI_MODE, _LIB, check_call
from .runtime_ctypes import DGLStreamHandle


def to_dgl_stream_handle(cuda_stream):
    """Convert torch.cuda.Stream to DGL stream handle

    Parameters
    ----------
    cuda_stream : torch.cuda.Stream.

    Returns
    -------
    DGLStreamHandle
        DGLStreamHandle of the input ``cuda_stream``.
    """
    return ctypes.c_void_p(cuda_stream.cuda_stream)


def _dgl_get_stream(ctx):
    """Get the current CUDA stream of the given DGL context.

    Parameters
    ----------
    ctx : DGL context.

    Returns
    -------
    DGLStreamHandle
        DGLStreamHandle of the current CUDA stream.
    """
    current_cuda_stream = DGLStreamHandle()
    check_call(
        _LIB.DGLGetStream(
            ctx.device_type, ctx.device_id, ctypes.byref(current_cuda_stream)
        )
    )
    return current_cuda_stream
