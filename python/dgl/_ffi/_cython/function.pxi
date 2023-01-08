import ctypes
import traceback
from cpython cimport Py_INCREF, Py_DECREF
from numbers import Number, Integral
from ..base import string_types
from ..object_generic import convert_to_object, ObjectGeneric
from ..runtime_ctypes import DGLDataType as CTypesDGLDataType, \
                             DGLContext as CTypesDGLContext, \
                             DGLByteArray


cdef void dgl_callback_finalize(void* fhandle):
    local_pyfunc = <object>(fhandle)
    Py_DECREF(local_pyfunc)

cdef int dgl_callback(DGLValue* args,
                      int* type_codes,
                      int num_args,
                      DGLRetValueHandle ret,
                      void* fhandle) with gil:
    cdef list pyargs
    cdef DGLValue value
    cdef int tcode
    local_pyfunc = <object>(fhandle)
    pyargs = []
    for i in range(num_args):
        value = args[i]
        tcode = type_codes[i]
        if (tcode == kObjectHandle or
            tcode == kFuncHandle or
            tcode == kModuleHandle or
            tcode > kExtBegin):
            CALL(DGLCbArgToReturn(&value, tcode))

        if tcode != kArrayHandle:
            pyargs.append(make_ret(value, tcode))
        else:
            pyargs.append(c_make_array(value.v_handle, True))
    try:
        rv = local_pyfunc(*pyargs)
    except Exception:
        msg = traceback.format_exc()
        DGLAPISetLastError(c_str(msg))
        return -1
    if rv is not None:
        if isinstance(rv, tuple):
            raise ValueError("PackedFunction can only support one return value")
        temp_args = []
        make_arg(rv, &value, &tcode, temp_args)
        CALL(DGLCFuncSetReturn(ret, &value, &tcode, 1))
    return 0


def convert_to_dgl_func(object pyfunc):
    """Convert a python function to DGL function

    Parameters
    ----------
    pyfunc : python function
        The python function to be converted.

    Returns
    -------
    dglfunc: dgl.Function
        The converted dgl function.
    """
    cdef DGLFunctionHandle chandle
    Py_INCREF(pyfunc)
    CALL(DGLFuncCreateFromCFunc(dgl_callback,
                                <void*>(pyfunc),
                                dgl_callback_finalize,
                                &chandle))
    ret = _CLASS_FUNCTION(None, False)
    (<FunctionBase>ret).chandle = chandle
    return ret


cdef inline int make_arg(object arg,
                         DGLValue* value,
                         int* tcode,
                         list temp_args) except -1:
    """Pack arguments into c args dgl call accept"""
    cdef unsigned long long ptr
    if isinstance(arg, ObjectBase):
        value[0].v_handle = (<ObjectBase>arg).chandle
        tcode[0] = kObjectHandle
    elif isinstance(arg, NDArrayBase):
        value[0].v_handle = (<NDArrayBase>arg).chandle
        tcode[0] = (kNDArrayContainer if
                    not (<NDArrayBase>arg).c_is_view else kArrayHandle)
    elif isinstance(arg, _DGL_COMPATS):
        ptr = arg._dgl_handle
        value[0].v_handle = (<void*>ptr)
        tcode[0] = arg.__class__._dgl_tcode
    elif isinstance(arg, (int, long)):
        value[0].v_int64 = arg
        tcode[0] = kObjectInt
    elif isinstance(arg, float):
        value[0].v_float64 = arg
        tcode[0] = kObjectFloat
    elif isinstance(arg, str):
        tstr = c_str(arg)
        value[0].v_str = tstr
        tcode[0] = kStr
        temp_args.append(tstr)
    elif arg is None:
        value[0].v_handle = NULL
        tcode[0] = kNull
    elif isinstance(arg, Number):
        value[0].v_float64 = arg
        tcode[0] = kObjectFloat
    elif isinstance(arg, CTypesDGLDataType):
        tstr = c_str(str(arg))
        value[0].v_str = tstr
        tcode[0] = kStr
        temp_args.append(tstr)
    elif isinstance(arg, CTypesDGLContext):
        value[0].v_ctx = (<DGLContext*>(
            <unsigned long long>ctypes.addressof(arg)))[0]
        tcode[0] = kDGLContext
    elif isinstance(arg, bytearray):
        arr = DGLByteArray()
        arr.data = ctypes.cast(
            (ctypes.c_byte * len(arg)).from_buffer(arg),
            ctypes.POINTER(ctypes.c_byte))
        arr.size = len(arg)
        value[0].v_handle = <void*>(
            <unsigned long long>ctypes.addressof(arr))
        tcode[0] = kBytes
        temp_args.append(arr)
    elif isinstance(arg, string_types):
        tstr = c_str(arg)
        value[0].v_str = tstr
        tcode[0] = kStr
        temp_args.append(tstr)
    elif isinstance(arg, (list, tuple, dict, ObjectGeneric)):
        arg = convert_to_object(arg)
        value[0].v_handle = (<ObjectBase>arg).chandle
        tcode[0] = kObjectHandle
        temp_args.append(arg)
    #elif isinstance(arg, _CLASS_MODULE):
    #    value[0].v_handle = c_handle(arg.handle)
    #    tcode[0] = kModuleHandle
    elif isinstance(arg, FunctionBase):
        value[0].v_handle = (<FunctionBase>arg).chandle
        tcode[0] = kFuncHandle
    elif isinstance(arg, ctypes.c_void_p):
        value[0].v_handle = c_handle(arg)
        tcode[0] = kHandle
    elif callable(arg):
        arg = convert_to_dgl_func(arg)
        value[0].v_handle = (<FunctionBase>arg).chandle
        tcode[0] = kFuncHandle
        temp_args.append(arg)
    else:
        raise TypeError("Don't know how to handle type %s" % type(arg))
    return 0

cdef inline bytearray make_ret_bytes(void* chandle):
    handle = ctypes_handle(chandle)
    arr = ctypes.cast(handle, ctypes.POINTER(DGLByteArray))[0]
    size = arr.size
    res = bytearray(size)
    rptr = (ctypes.c_byte * size).from_buffer(res)
    if not ctypes.memmove(rptr, arr.data, size):
        raise RuntimeError('memmove failed')
    return res

cdef inline object make_ret(DGLValue value, int tcode):
    """convert result to return value."""
    if tcode == kObjectHandle:
        return make_ret_object(value.v_handle)
    elif tcode == kNull:
        return None
    elif tcode == kObjectInt:
        return value.v_int64
    elif tcode == kObjectFloat:
        return value.v_float64
    elif tcode == kNDArrayContainer:
        return c_make_array(value.v_handle, False)
    elif tcode == kStr:
        return py_str(value.v_str)
    elif tcode == kBytes:
        return make_ret_bytes(value.v_handle)
    elif tcode == kHandle:
        return ctypes_handle(value.v_handle)
    elif tcode == kDGLContext:
        return CTypesDGLContext(value.v_ctx.device_type, value.v_ctx.device_id)
    # (minjie): class module are not used in DGL.
    #elif tcode == kModuleHandle:
    #    return _CLASS_MODULE(ctypes_handle(value.v_handle))
    elif tcode == kFuncHandle:
        fobj = _CLASS_FUNCTION(None, False)
        (<FunctionBase>fobj).chandle = value.v_handle
        return fobj
    elif tcode in _DGL_EXT_RET:
        return _DGL_EXT_RET[tcode](ctypes_handle(value.v_handle))

    raise ValueError("Unhandled type code %d" % tcode)


cdef inline int FuncCall3(void* chandle,
                          tuple args,
                          int nargs,
                          DGLValue* ret_val,
                          int* ret_tcode) except -1:
    cdef DGLValue[3] values
    cdef int[3] tcodes
    nargs = len(args)
    temp_args = []
    for i in range(nargs):
        make_arg(args[i], &values[i], &tcodes[i], temp_args)
    with nogil:
        ret = DGLFuncCall(chandle, &values[0], &tcodes[0],
                          nargs, ret_val, ret_tcode)
    if ret != 0:
        raise DGLError(py_str(DGLGetLastError()))
    return 0

cdef inline int FuncCall(void* chandle,
                         tuple args,
                         DGLValue* ret_val,
                         int* ret_tcode) except -1:
    cdef int nargs
    nargs = len(args)
    if nargs <= 3:
        FuncCall3(chandle, args, nargs, ret_val, ret_tcode)
        return 0

    cdef vector[DGLValue] values
    cdef vector[int] tcodes
    values.resize(max(nargs, 1))
    tcodes.resize(max(nargs, 1))
    temp_args = []
    for i in range(nargs):
        make_arg(args[i], &values[i], &tcodes[i], temp_args)
    with nogil:
        ret = DGLFuncCall(chandle, &values[0], &tcodes[0],
                          nargs, ret_val, ret_tcode)
    if ret != 0:
        raise DGLError(py_str(DGLGetLastError()))
    return 0


cdef inline int ConstructorCall(void* constructor_handle,
                                int type_code,
                                tuple args,
                                void** handle) except -1:
    """Call contructor of a handle function"""
    cdef DGLValue ret_val
    cdef int ret_tcode
    FuncCall(constructor_handle, args, &ret_val, &ret_tcode)
    assert ret_tcode == type_code
    handle[0] = ret_val.v_handle
    return 0


cdef class FunctionBase:
    cdef DGLFunctionHandle chandle
    cdef int is_global

    cdef inline _set_handle(self, handle):
        if handle is None:
            self.chandle = NULL
        else:
            self.chandle = c_handle(handle)

    property is_global:
        def __get__(self):
            return self.c_is_global != 0

        def __set__(self, value):
            self.c_is_global = value

    property handle:
        def __get__(self):
            if self.chandle == NULL:
                return None
            else:
                return ctypes.cast(<unsigned long long>self.chandle, ctypes.c_void_p)
        def __set__(self, value):
            self._set_handle(value)

    def __init__(self, handle, is_global):
        self._set_handle(handle)
        self.c_is_global = is_global

    def __dealloc__(self):
        if self.is_global == 0:
            CALL(DGLFuncFree(self.chandle))

    def __call__(self, *args):
        cdef DGLValue ret_val
        cdef int ret_tcode
        FuncCall(self.chandle, args, &ret_val, &ret_tcode)
        return make_ret(ret_val, ret_tcode)

_CLASS_FUNCTION = None
_CLASS_MODULE = None

def _set_class_module(module_class):
    """Initialize the module."""
    global _CLASS_MODULE
    _CLASS_MODULE = module_class

def _set_class_function(func_class):
    global _CLASS_FUNCTION
    _CLASS_FUNCTION = func_class
