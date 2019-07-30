"""The C Types used in API."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs

import ctypes
from ..base import py_str, check_call, _LIB
from ..runtime_ctypes import DGLByteArray, TypeCode, DGLType, DGLContext

class DGLValue(ctypes.Union):
    """DGLValue in C API"""
    _fields_ = [("v_int64", ctypes.c_int64),
                ("v_float64", ctypes.c_double),
                ("v_handle", ctypes.c_void_p),
                ("v_str", ctypes.c_char_p),
                ("v_type", DGLType),
                ("v_ctx", DGLContext)]


DGLPackedCFunc = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(DGLValue),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p)


DGLCFuncFinalizer = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p)


def _return_handle(x):
    """return handle"""
    handle = x.v_handle
    if not isinstance(handle, ctypes.c_void_p):
        handle = ctypes.c_void_p(handle)
    return handle

def _return_bytes(x):
    """return handle"""
    handle = x.v_handle
    if not isinstance(handle, ctypes.c_void_p):
        handle = ctypes.c_void_p(handle)
    arr = ctypes.cast(handle, ctypes.POINTER(DGLByteArray))[0]
    size = arr.size
    res = bytearray(size)
    rptr = (ctypes.c_byte * size).from_buffer(res)
    if not ctypes.memmove(rptr, arr.data, size):
        raise RuntimeError('memmove failed')
    return res

def _wrap_arg_func(return_f, type_code):
    tcode = ctypes.c_int(type_code)
    def _wrap_func(x):
        check_call(_LIB.DGLCbArgToReturn(ctypes.byref(x), tcode))
        return return_f(x)
    return _wrap_func

RETURN_SWITCH = {
    TypeCode.INT: lambda x: x.v_int64,
    TypeCode.FLOAT: lambda x: x.v_float64,
    TypeCode.HANDLE: _return_handle,
    TypeCode.NULL: lambda x: None,
    TypeCode.STR: lambda x: py_str(x.v_str),
    TypeCode.BYTES: _return_bytes,
    TypeCode.DGL_CONTEXT: lambda x: DGLContext(x.v_ctx.device_type, x.v_ctx.device_id),
}

C_TO_PY_ARG_SWITCH = {
    TypeCode.INT: lambda x: x.v_int64,
    TypeCode.FLOAT: lambda x: x.v_float64,
    TypeCode.HANDLE: _return_handle,
    TypeCode.NULL: lambda x: None,
    TypeCode.STR: lambda x: py_str(x.v_str),
    TypeCode.BYTES: _return_bytes,
    TypeCode.DGL_CONTEXT: lambda x: DGLContext(x.v_ctx.device_type, x.v_ctx.device_id),
}
