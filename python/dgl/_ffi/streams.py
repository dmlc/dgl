# pylint: disable=invalid-name, unused-import
"""Runtime stream api which is maily for internal use only."""
from __future__ import absolute_import

import ctypes
from .base import _LIB, check_call, _FFI_MODE
from .runtime_ctypes import DGLStreamHandle


IMPORT_EXCEPT = RuntimeError if _FFI_MODE == "cython" else ImportError


def to_dgl_stream_handle(stream):
    """ Convert torch.cuda.Stream to DGL stream handle

    Parameters
    ----------
    stream : torch.cuda.Stream.

    Returns
    -------
    DGLStreamHandle.
    """
    return ctypes.c_void_p(stream.cuda_stream)

def _dgl_set_stream(ctx, stream):
    """ Set the current CUDA stream of the given DGL context.

    Parameters
    ----------
    ctx: DGLContext
    stream: DGLStreamHandle
    """
    check_call(_LIB.DGLSetStream(ctx.device_type, ctx.device_id, stream))

def _dgl_get_stream(ctx):
    """Get the current CUDA stream of the given DGL context.

    Parameters
    ----------
    ctx : DGL context.

    Returns
    -------
    DGLStreamHandle.
    """
    current_cuda_stream = DGLStreamHandle()
    check_call(_LIB.DGLGetStream(
        ctx.device_type, ctx.device_id, ctypes.byref(current_cuda_stream)))
    return current_cuda_stream
