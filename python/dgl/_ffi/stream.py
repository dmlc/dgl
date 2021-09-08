# pylint: disable=invalid-name, unused-import
"""Runtime stream api which is maily for internal use only."""
from __future__ import absolute_import

import ctypes
from .base import _LIB, check_call, _FFI_MODE
from .runtime_ctypes import DGLStreamHandle
from .ndarray import context


IMPORT_EXCEPT = RuntimeError if _FFI_MODE == "cython" else ImportError


class StreamBase(object):
    """ A stream base class which wraps cuda stream and corresponding context. """

    def __init__(self, ctx, stream):
        """ init
        Parameters
        ----------
        ctx : DGLContext
            context on which the stream is.
        stream : DGLStreamHandle
            a pointer which could be converted into cudaStream_t directly.
        """
        self.ctx = ctx
        self.stream = stream


class Stream(StreamBase):
    """ A wrapper of internal stream. It's created/freed via underlying DGL stream calls. """

    def __init__(self, ctx=None):
        """ create stream
        Parameters
        ----------
        ctx : DGLContext
            context on which the stream lies.
        """
        if ctx is None:
            ctx = context('cuda')
        stream = DGLStreamHandle()
        check_call(_LIB.DGLStreamCreate(
            ctx.device_type, ctx.device_id, ctypes.byref(stream)))
        super().__init__(ctx, stream)

    def __del__(self):
        """ free created stream
        """
        ctx = self.ctx
        check_call(_LIB.DGLStreamFree(
            ctx.device_type, ctx.device_id, self.stream))


class ExternalStream(StreamBase):
    """ A wrapper of external stream. As the stream is created externally,
        it's user's responsibility to keep the referenced stream alive while
        this class is being used.
    """


def create_stream(ctx=None):
    """ create a stream via DGL stream calls.

    Parameters
    ----------
    ctx : DGLContext
        context on which the stream is.

    Returns
    -------
    stream : Stream
        a stream wrapper.
    """
    return Stream(ctx)


def set_stream(stream):
    """ set a stream as current via DGL stream calls.

    Parameters
    ----------
    stream : StreamBase
        a stream wrapper which includes context and underlying stream.

    Returns
    -------
    """
    ctx = stream.ctx
    check_call(_LIB.DGLSetStream(
        ctx.device_type, ctx.device_id, stream.stream))


def current_stream(ctx):
    """ get current stream via DGL stream calls.

    Parameters
    ----------
    ctx : DGLContext
        context on which the stream is.

    Returns
    -------
    stream : StreamBase
        a stream wrapper.
    """
    stream = DGLStreamHandle()
    check_call(_LIB.DGLGetStream(
        ctx.device_type, ctx.device_id, ctypes.byref(stream)))
    return StreamBase(ctx, stream)


def synchronize_stream(stream):
    """ synchronize stream via DGL stream calls.

    Parameters
    ----------
    stream : StreamBase
        target stream to be synchronized.

    Returns
    -------
    """
    ctx = stream.ctx
    check_call(_LIB.DGLSynchronize(
        ctx.device_type, ctx.device_id, stream.stream))
