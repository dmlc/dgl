# pylint: disable=invalid-name, unused-import
"""Runtime stream api which is maily for internal use only."""
from __future__ import absolute_import

import ctypes
from .base import _LIB, check_call, _FFI_MODE
from .runtime_ctypes import DGLStreamHandle
from .ndarray import context
from ..utils import to_dgl_context


IMPORT_EXCEPT = RuntimeError if _FFI_MODE == "cython" else ImportError


class StreamContext(object):
    """ Context-manager that selects a given stream.

    All CUDA kernels queued within its context will be enqueued
    on a selected stream.

    """

    def __init__(self, cuda_stream):
        """ create stream context instance

        Parameters
        ----------
        cuda_stream : torch.cuda.Stream. This manager is a no-op if it's ``None``.
            target stream will be set.
        """
        if cuda_stream is None:
            self.curr_cuda_stream = None
        else:
            self.ctx = to_dgl_context(cuda_stream.device)
            self.curr_cuda_stream = cuda_stream.cuda_stream

    def __enter__(self):
        """ get previous stream and set target stream as current.
        """
        if self.curr_cuda_stream is None:
            return
        self.prev_cuda_stream = DGLStreamHandle()
        check_call(_LIB.DGLGetStream(
            self.ctx.device_type, self.ctx.device_id, ctypes.byref(self.prev_cuda_stream)))
        check_call(_LIB.DGLSetStream(
            self.ctx.device_type, self.ctx.device_id, ctypes.c_void_p(self.curr_cuda_stream)))

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ restore previous stream when exiting.
        """
        if self.curr_cuda_stream is None:
            return
        check_call(_LIB.DGLSetStream(
            self.ctx.device_type, self.ctx.device_id, self.prev_cuda_stream))


def stream(cuda_stream):
    """ Wrapper of StreamContext

    Parameters
    ----------
    stream : torch.cuda.Stream. This manager is a no-op if it's ``None``.
        target stream will be set.
    """
    return StreamContext(cuda_stream)

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

def set_stream(stream):
    """ Set the current CUDA stream of DGL

    Parameters
    ----------
    stream : torch.cuda.Stream.
    """
    if stream is None:
        return

    ctx = to_dgl_context(stream.device)
    check_call(_LIB.DGLSetStream(
        ctx.device_type, ctx.device_id, ctypes.c_void_p(stream.cuda_stream)))

def get_current_stream(ctx):
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
