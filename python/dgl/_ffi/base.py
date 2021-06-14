# coding: utf-8
# pylint: disable=invalid-name
"""ctypes library and helper functions """
from __future__ import absolute_import

import sys
import os
import ctypes
import numpy as np
from . import libinfo

#----------------------------
# library loading
#----------------------------
if sys.version_info[0] == 3:
    string_types = (str,)
    numeric_types = (float, int, np.float32, np.int32)
    # this function is needed for python3
    # to convert ctypes.char_p .value back to python str
    py_str = lambda x: x.decode('utf-8')
else:
    string_types = (basestring,)
    numeric_types = (float, int, long, np.float32, np.int32)
    py_str = lambda x: x


class DGLError(Exception):
    """Error thrown by DGL function"""
    pass  # pylint: disable=unnecessary-pass

def _load_lib():
    """Load libary by searching possible path."""
    lib_path = libinfo.find_lib_path()
    lib = ctypes.CDLL(lib_path[0])
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

#----------------------------
# helper function in ctypes.
#----------------------------
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
    return ctypes.c_char_p(string.encode('utf-8'))


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


def load_tensor_adapter(backend, version):
    """Tell DGL to load a tensoradapter library for given backend and version.

    Parameters
    ----------
    backend : str
        The backend (currently ``pytorch``, ``mxnet`` or ``tensorflow``).
    version : str
        The version number of the backend.
    """
    version = version.split('+')[0]
    if sys.platform.startswith('linux'):
        basename = 'libtensoradapter_%s_%s.so' % (backend, version)
    elif sys.platform.startswith('darwin'):
        basename = 'libtensoradapter_%s_%s.dylib' % (backend, version)
    elif sys.platform.startswith('win'):
        basename = 'tensoradapter_%s_%s.dll' % (backend, version)
    else:
        raise NotImplementedError('Unsupported system: %s' % sys.platform)
    path = os.path.join(_DIR_NAME, 'tensoradapter', backend, basename)
    _LIB.DGLLoadTensorAdapter(path.encode('utf-8'))
