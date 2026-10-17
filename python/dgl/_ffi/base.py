# coding: utf-8
# pylint: disable=invalid-name
"""ctypes library and helper functions """
from __future__ import absolute_import

import ctypes
import logging
import os
import sys

import numpy as np

from . import libinfo

# ----------------------------
# library loading
# ----------------------------
if sys.version_info[0] == 3:
    string_types = (str,)
    numeric_types = (float, int, np.float32, np.int32)
    # this function is needed for python3
    # to convert ctypes.char_p .value back to python str
    py_str = lambda x: x.decode("utf-8")
else:
    string_types = (basestring,)
    numeric_types = (float, int, long, np.float32, np.int32)
    py_str = lambda x: x


class DGLError(Exception):
    """Error thrown by DGL function"""

    pass  # pylint: disable=unnecessary-pass


# Latch state for the torch_npu <-> DGL Ascend stream bridge.
_ascend_bridge = {"ok": False}

torch = None


def _bridge_ascend_stream():
    """Synchronize torch_npu's current stream before entering libdgl.

    torch_npu submits kernels asynchronously on its own (non-default) ACL
    stream, while DGL Ascend kernels run on the ACL default stream and block
    on it until completion.  Without a barrier, a torch_npu write (e.g. a
    feature tensor copied to NPU) may still be in flight when a DGL kernel
    reads it, producing nondeterministic results or device-side faults.

    Directly handing torch's stream to DGL (DGLSetStream) is not safe either:
    torch_npu submits work to that stream from its internal task-queue
    thread, while DGL would submit from the caller thread - concurrent
    submissions on one stream corrupt the driver state.

    The simplest correct barrier is therefore to drain torch_npu's current
    stream on the host before each DGL C API call.  DGL ops themselves
    synchronize at the end, so the reverse (DGL -> torch) direction is
    already ordered.  When the torch stream is idle this call is cheap.

    Set DGL_ASCEND_STREAM_BRIDGE=0 to disable (for debugging only).
    """
    global torch
    if os.environ.get("DGL_ASCEND_STREAM_BRIDGE", "1") == "0":
        return
    if sys.is_finalizing():
        # Interpreter shutdown: destructors must not touch torch_npu.
        return
    st = _ascend_bridge
    if not st["ok"]:
        # torch_npu may be imported after dgl; retry cheaply until latched.
        if "torch_npu" not in sys.modules:
            return
        try:
            import torch as _torch
            import torch_npu  # noqa: F401

            torch = _torch
            if not torch.npu.is_available():
                return
            st["ok"] = True
        except Exception:  # pylint: disable=broad-except
            return
    torch.npu.current_stream().synchronize()


class _AscendStreamFuncWrapper(object):
    """Wraps a libdgl C function pointer to bridge the stream before each call.

    Attribute access (e.g. ``restype``) is forwarded to the underlying
    function pointer.
    """

    __slots__ = ("_func",)

    def __init__(self, func):
        object.__setattr__(self, "_func", func)

    def __call__(self, *args, **kwargs):
        _bridge_ascend_stream()
        return self._func(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._func, name)

    def __setattr__(self, name, value):
        setattr(self._func, name, value)


class _SyncedCDLL(ctypes.CDLL):
    """CDLL whose function calls first bridge the torch_npu stream."""

    def __getattr__(self, name):
        func = super(_SyncedCDLL, self).__getattr__(name)
        wrapped = _AscendStreamFuncWrapper(func)
        self.__dict__[name] = wrapped
        return wrapped


def _load_lib():
    """Load libary by searching possible path."""
    lib_path = libinfo.find_lib_path()
    lib = _SyncedCDLL(lib_path[0])
    dirname = os.path.dirname(lib_path[0])
    basename = os.path.basename(lib_path[0])
    # DMatrix functions
    lib.DGLGetLastError.restype = ctypes.c_char_p
    return lib, basename, dirname


# version number
__version__ = libinfo.__version__
# library instance of nnvm
_LIB, _LIB_NAME, _DIR_NAME = _load_lib()

# The FFI mode of DGL
_FFI_MODE = os.environ.get("DGL_FFI", "auto")

# ----------------------------
# helper function in ctypes.
# ----------------------------
def check_call(ret):
    """Check the return value of C API call

    This function will raise exception when error occurs.
    Wrap every API call with this function

    Parameters
    ----------
    ret : int
        return value from API calls
    """
    if ret != 0:
        raise DGLError(py_str(_LIB.DGLGetLastError()))


def c_str(string):
    """Create ctypes char * from a python string
    Parameters
    ----------
    string : string type
        python string

    Returns
    -------
    str : c_char_p
        A char pointer that can be passed to C API
    """
    return ctypes.c_char_p(string.encode("utf-8"))


def c_array(ctype, values):
    """Create ctypes array from a python array

    Parameters
    ----------
    ctype : ctypes data type
        data type of the array we want to convert to

    values : tuple or list
        data content

    Returns
    -------
    out : ctypes array
        Created ctypes array
    """
    return (ctype * len(values))(*values)


def decorate(func, fwrapped):
    """A wrapper call of decorator package, differs to call time

    Parameters
    ----------
    func : function
        The original function

    fwrapped : function
        The wrapped function
    """
    import decorator

    return decorator.decorate(func, fwrapped)


tensor_adapter_loaded = False


def load_tensor_adapter(backend, version):
    """Tell DGL to load a tensoradapter library for given backend and version.

    Parameters
    ----------
    backend : str
        The backend (currently ``pytorch``, ``mxnet`` or ``tensorflow``).
    version : str
        The version number of the backend.
    """
    global tensor_adapter_loaded
    version = version.split("+")[0]
    if sys.platform.startswith("linux"):
        basename = "libtensoradapter_%s_%s.so" % (backend, version)
    elif sys.platform.startswith("darwin"):
        basename = "libtensoradapter_%s_%s.dylib" % (backend, version)
    elif sys.platform.startswith("win"):
        basename = "tensoradapter_%s_%s.dll" % (backend, version)
    else:
        raise NotImplementedError("Unsupported system: %s" % sys.platform)
    path = os.path.join(_DIR_NAME, "tensoradapter", backend, basename)
    tensor_adapter_loaded = _LIB.DGLLoadTensorAdapter(path.encode("utf-8")) == 0
    if not tensor_adapter_loaded:
        logger = logging.getLogger("dgl-core")
        logger.debug("Memory optimization with PyTorch is not enabled.")


def is_tensor_adaptor_enabled() -> bool:
    """Check whether TensorAdaptor is enabled."""
    return tensor_adapter_loaded
