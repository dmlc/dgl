# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-branches, global-statement
"""Function configuration API."""
from __future__ import absolute_import

import ctypes
import traceback
from numbers import Number, Integral

from ..base import _LIB, check_call
from ..base import c_str, string_types
from ..object_generic import convert_to_object, ObjectGeneric
from ..runtime_ctypes import DGLType, DGLByteArray, DGLContext
from . import ndarray as _nd
from .ndarray import NDArrayBase, _make_array
from .types import DGLValue, TypeCode
from .types import DGLPackedCFunc, DGLCFuncFinalizer
from .types import RETURN_SWITCH, C_TO_PY_ARG_SWITCH, _wrap_arg_func
from .object import ObjectBase
from . import object as _object

FunctionHandle = ctypes.c_void_p
ModuleHandle = ctypes.c_void_p
DGLRetValueHandle = ctypes.c_void_p

def _ctypes_free_resource(rhandle):
    """callback to free resources when it it not needed."""
    pyobj = ctypes.cast(rhandle, ctypes.py_object)
    ctypes.pythonapi.Py_DecRef(pyobj)

# Global callback that is always alive
DGL_FREE_PYOBJ = DGLCFuncFinalizer(_ctypes_free_resource)
ctypes.pythonapi.Py_IncRef(ctypes.py_object(DGL_FREE_PYOBJ))

def convert_to_dgl_func(pyfunc):
    """Convert a python function to DGL function

    Parameters
    ----------
    pyfunc : python function
        The python function to be converted.

    Returns
    -------
    dglfunc: dgl.nd.Function
        The converted dgl function.
    """
    local_pyfunc = pyfunc
    def cfun(args, type_codes, num_args, ret, _):
        """ ctypes function """
        num_args = num_args.value if isinstance(num_args, ctypes.c_int) else num_args
        pyargs = (C_TO_PY_ARG_SWITCH[type_codes[i]](args[i]) for i in range(num_args))
        # pylint: disable=broad-except
        try:
            rv = local_pyfunc(*pyargs)
        except Exception:
            msg = traceback.format_exc()
            _LIB.DGLAPISetLastError(c_str(msg))
            return -1

        if rv is not None:
            if isinstance(rv, tuple):
                raise ValueError("PackedFunction can only support one return value")
            temp_args = []
            values, tcodes, _ = _make_dgl_args((rv,), temp_args)
            if not isinstance(ret, DGLRetValueHandle):
                ret = DGLRetValueHandle(ret)
            check_call(_LIB.DGLCFuncSetReturn(ret, values, tcodes, ctypes.c_int(1)))
            _ = temp_args
            _ = rv
        return 0

    handle = FunctionHandle()
    f = DGLPackedCFunc(cfun)
    # NOTE: We will need to use python-api to increase ref count of the f
    # DGL_FREE_PYOBJ will be called after it is no longer needed.
    pyobj = ctypes.py_object(f)
    ctypes.pythonapi.Py_IncRef(pyobj)
    check_call(_LIB.DGLFuncCreateFromCFunc(
        f, pyobj, DGL_FREE_PYOBJ, ctypes.byref(handle)))
    return _CLASS_FUNCTION(handle, False)


def _make_dgl_args(args, temp_args):
    """Pack arguments into c args dgl call accept.

    temp_args is used to temporarily save the arguments so they will not be
    freed during C API function call.
    """
    num_args = len(args)
    values = (DGLValue * num_args)()
    type_codes = (ctypes.c_int * num_args)()
    for i, arg in enumerate(args):
        if arg is None:
            values[i].v_handle = None
            type_codes[i] = TypeCode.NULL
        elif isinstance(arg, ObjectBase):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.OBJECT_HANDLE
        elif isinstance(arg, (list, tuple, dict, ObjectGeneric)):
            arg = convert_to_object(arg)
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.OBJECT_HANDLE
            temp_args.append(arg)
        elif isinstance(arg, NDArrayBase):
            values[i].v_handle = ctypes.cast(arg.handle, ctypes.c_void_p)
            type_codes[i] = (TypeCode.NDARRAY_CONTAINER
                             if not arg.is_view else TypeCode.ARRAY_HANDLE)
        elif isinstance(arg, _nd._DGL_COMPATS):
            values[i].v_handle = ctypes.c_void_p(arg._dgl_handle)
            type_codes[i] = arg.__class__._dgl_tcode
        elif isinstance(arg, Integral):
            values[i].v_int64 = arg
            type_codes[i] = TypeCode.INT
        elif isinstance(arg, Number):
            values[i].v_float64 = arg
            type_codes[i] = TypeCode.FLOAT
        elif isinstance(arg, DGLType):
            values[i].v_str = c_str(str(arg))
            type_codes[i] = TypeCode.STR
        elif isinstance(arg, DGLContext):
            values[i].v_ctx = arg
            type_codes[i] = TypeCode.DGL_CONTEXT
        elif isinstance(arg, bytearray):
            arr = DGLByteArray()
            arr.data = ctypes.cast(
                (ctypes.c_byte * len(arg)).from_buffer(arg),
                ctypes.POINTER(ctypes.c_byte))
            arr.size = len(arg)
            values[i].v_handle = ctypes.c_void_p(ctypes.addressof(arr))
            temp_args.append(arr)
            type_codes[i] = TypeCode.BYTES
        elif isinstance(arg, string_types):
            values[i].v_str = c_str(arg)
            type_codes[i] = TypeCode.STR
        # NOTE(minjie): module is not used in DGL
        #elif isinstance(arg, _CLASS_MODULE):
        #    values[i].v_handle = arg.handle
        #    type_codes[i] = TypeCode.MODULE_HANDLE
        elif isinstance(arg, FunctionBase):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.FUNC_HANDLE
        elif isinstance(arg, ctypes.c_void_p):
            values[i].v_handle = arg
            type_codes[i] = TypeCode.HANDLE
        elif callable(arg):
            arg = convert_to_dgl_func(arg)
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.FUNC_HANDLE
            temp_args.append(arg)
        else:
            raise TypeError("Don't know how to handle type %s" % type(arg))
    return values, type_codes, num_args


class FunctionBase(object):
    """Function base."""
    __slots__ = ["handle", "is_global"]
    # pylint: disable=no-member
    def __init__(self, handle, is_global):
        """Initialize the function with handle

        Parameters
        ----------
        handle : FunctionHandle
            the handle to the underlying function.

        is_global : bool
            Whether this is a global function in python
        """
        self.handle = handle
        self.is_global = is_global

    def __del__(self):
        if not self.is_global and _LIB is not None:
            check_call(_LIB.DGLFuncFree(self.handle))

    def __call__(self, *args):
        """Call the function with positional arguments

        args : list
           The positional arguments to the function call.
        """
        temp_args = []
        values, tcodes, num_args = _make_dgl_args(args, temp_args)
        ret_val = DGLValue()
        ret_tcode = ctypes.c_int()
        check_call(_LIB.DGLFuncCall(
            self.handle, values, tcodes, ctypes.c_int(num_args),
            ctypes.byref(ret_val), ctypes.byref(ret_tcode)))
        _ = temp_args
        _ = args
        return RETURN_SWITCH[ret_tcode.value](ret_val)


def __init_handle_by_constructor__(fconstructor, args):
    """Initialize handle by constructor"""
    temp_args = []
    values, tcodes, num_args = _make_dgl_args(args, temp_args)
    ret_val = DGLValue()
    ret_tcode = ctypes.c_int()
    check_call(_LIB.DGLFuncCall(
        fconstructor.handle, values, tcodes, ctypes.c_int(num_args),
        ctypes.byref(ret_val), ctypes.byref(ret_tcode)))
    _ = temp_args
    _ = args
    assert ret_tcode.value == TypeCode.OBJECT_HANDLE
    handle = ret_val.v_handle
    return handle


def _return_module(x):
    """Return function"""
    handle = x.v_handle
    if not isinstance(handle, ModuleHandle):
        handle = ModuleHandle(handle)
    return _CLASS_MODULE(handle)

def _handle_return_func(x):
    """Return function"""
    handle = x.v_handle
    if not isinstance(handle, FunctionHandle):
        handle = FunctionHandle(handle)
    return _CLASS_FUNCTION(handle, False)


# setup return handle for function type
_object.__init_by_constructor__ = __init_handle_by_constructor__
RETURN_SWITCH[TypeCode.FUNC_HANDLE] = _handle_return_func
RETURN_SWITCH[TypeCode.MODULE_HANDLE] = _return_module
RETURN_SWITCH[TypeCode.NDARRAY_CONTAINER] = lambda x: _make_array(x.v_handle, False)
C_TO_PY_ARG_SWITCH[TypeCode.FUNC_HANDLE] = _wrap_arg_func(
    _handle_return_func, TypeCode.FUNC_HANDLE)
C_TO_PY_ARG_SWITCH[TypeCode.MODULE_HANDLE] = _wrap_arg_func(
    _return_module, TypeCode.MODULE_HANDLE)
C_TO_PY_ARG_SWITCH[TypeCode.ARRAY_HANDLE] = lambda x: _make_array(x.v_handle, True)
C_TO_PY_ARG_SWITCH[TypeCode.NDARRAY_CONTAINER] = lambda x: _make_array(x.v_handle, False)

_CLASS_MODULE = None
_CLASS_FUNCTION = None

def _set_class_module(module_class):
    """Initialize the module."""
    global _CLASS_MODULE
    _CLASS_MODULE = module_class

def _set_class_function(func_class):
    global _CLASS_FUNCTION
    _CLASS_FUNCTION = func_class
