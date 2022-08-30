from .._ffi.streams import to_dgl_stream_handle, \
                           _dgl_get_stream, _dgl_set_stream
from .._ffi.runtime_ctypes import DGLContext
from ..utils import to_dgl_context

__all__ = [
    'to_dgl_stream_handle',
    '_dgl_get_stream',
    '_dgl_set_stream',
    'set_stream',
    'current_stream',
    'StreamContext',
    'stream'
]

def set_stream(stream):
    """ Set the current CUDA stream of DGL.
    Usage of this function is discouraged in favor of the ``stream`` context manager.

    Parameters
    ----------
    stream : torch.cuda.Stream.
    """
    if stream is None:
        return

    ctx = to_dgl_context(stream.device)
    _dgl_set_stream(ctx, to_dgl_stream_handle(stream))

def current_stream(ctx):
    """Get the current CUDA stream of a given context.

    Parameters
    ----------
    ctx : DGLContext or backend context.

    Note
    _________
    The returned DGLStreamHandle object cannot be used as an argument for ``set_stream()``.
    Please use ``_dgl_set_stream(DGLContext, DGLStreamHandle)`` instead.

    Returns
    -------
    DGLStreamHandle.
    """
    if not isinstance(ctx, DGLContext):
        ctx = to_dgl_context(ctx)
    return _dgl_get_stream(ctx)

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
            self.curr_cuda_stream = cuda_stream

    def __enter__(self):
        """ get previous stream and set target stream as current.
        """
        if self.curr_cuda_stream is None:
            return
        self.prev_cuda_stream = current_stream(self.ctx)
        set_stream(self.curr_cuda_stream)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ restore previous stream when exiting.
        """
        if self.curr_cuda_stream is None:
            return
        _dgl_set_stream(self.ctx, self.prev_cuda_stream)

def stream(cuda_stream):
    """ Wrapper of StreamContext

    Parameters
    ----------
    stream : torch.cuda.Stream. This manager is a no-op if it's ``None``.
        target stream will be set.
    """
    return StreamContext(cuda_stream)
