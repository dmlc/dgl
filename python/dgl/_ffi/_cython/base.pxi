from ..base import DGLError
from libcpp.vector cimport vector
from cpython.version cimport PY_MAJOR_VERSION
from cpython cimport pycapsule
from libc.stdint cimport int64_t, uint64_t, uint8_t, uint16_t
import ctypes

cdef enum TVMTypeCode:
    kInt = 0
    kUInt = 1
    kFloat = 2
    kHandle = 3
    kNull = 4
    kTVMType = 5
    kTVMContext = 6
    kArrayHandle = 7
    kNodeHandle = 8
    kModuleHandle = 9
    kFuncHandle = 10
    kStr = 11
    kBytes = 12
    kNDArrayContainer = 13
    kExtBegin = 15

cdef extern from "tvm/runtime/c_runtime_api.h":
    ctypedef struct DLDataType:
        uint8_t code
        uint8_t bits
        uint16_t lanes

    ctypedef struct DLContext:
        int device_type
        int device_id

    ctypedef struct DLTensor:
        void* data
        DLContext ctx
        int ndim
        DLDataType dtype
        int64_t* shape
        int64_t* strides
        uint64_t byte_offset

    ctypedef struct DLManagedTensor:
        DLTensor dl_tensor
        void* manager_ctx
        void (*deleter)(DLManagedTensor* self)

    ctypedef struct TVMValue:
        int64_t v_int64
        double v_float64
        void* v_handle
        const char* v_str
        DLDataType v_type
        DLContext v_ctx

ctypedef int64_t tvm_index_t
ctypedef DLTensor* DLTensorHandle
ctypedef void* TVMStreamHandle
ctypedef void* TVMRetValueHandle
ctypedef void* TVMFunctionHandle
ctypedef void* NodeHandle

ctypedef int (*TVMPackedCFunc)(
    TVMValue* args,
    int* type_codes,
    int num_args,
    TVMRetValueHandle ret,
    void* resource_handle)

ctypedef void (*TVMPackedCFuncFinalizer)(void* resource_handle)

cdef extern from "tvm/runtime/c_runtime_api.h":
    void TVMAPISetLastError(const char* msg)
    const char *TVMGetLastError()
    int TVMFuncCall(TVMFunctionHandle func,
                    TVMValue* arg_values,
                    int* type_codes,
                    int num_args,
                    TVMValue* ret_val,
                    int* ret_type_code)
    int TVMFuncFree(TVMFunctionHandle func)
    int TVMCFuncSetReturn(TVMRetValueHandle ret,
                          TVMValue* value,
                          int* type_code,
                          int num_ret)
    int TVMFuncCreateFromCFunc(TVMPackedCFunc func,
                               void* resource_handle,
                               TVMPackedCFuncFinalizer fin,
                               TVMFunctionHandle *out)
    int TVMCbArgToReturn(TVMValue* value, int code)
    int TVMArrayAlloc(tvm_index_t* shape,
                      tvm_index_t ndim,
                      DLDataType dtype,
                      DLContext ctx,
                      DLTensorHandle* out)
    int TVMArrayFree(DLTensorHandle handle)
    int TVMArrayCopyFromTo(DLTensorHandle src,
                           DLTensorHandle to,
                           TVMStreamHandle stream)
    int TVMArrayFromDLPack(DLManagedTensor* arr_from,
                           DLTensorHandle* out)
    int TVMArrayToDLPack(DLTensorHandle arr_from,
                         DLManagedTensor** out)
    void TVMDLManagedTensorCallDeleter(DLManagedTensor* dltensor)

cdef extern from "tvm/c_dsl_api.h":
    int TVMNodeFree(NodeHandle handle)
    int TVMNodeTypeKey2Index(const char* type_key,
                             int* out_index)
    int TVMNodeGetTypeIndex(NodeHandle handle,
                            int* out_index)
    int TVMNodeGetAttr(NodeHandle handle,
                       const char* key,
                       TVMValue* out_value,
                       int* out_type_code,
                       int* out_success)

cdef inline py_str(const char* x):
    if PY_MAJOR_VERSION < 3:
        return x
    else:
        return x.decode("utf-8")


cdef inline c_str(pystr):
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
    return pystr.encode("utf-8")


cdef inline CALL(int ret):
    if ret != 0:
        raise DGLError(py_str(TVMGetLastError()))


cdef inline object ctypes_handle(void* chandle):
    """Cast C handle to ctypes handle."""
    return ctypes.cast(<unsigned long long>chandle, ctypes.c_void_p)


cdef inline void* c_handle(object handle):
    """Cast C types handle to c handle."""
    cdef unsigned long long v_ptr
    v_ptr = handle.value
    return <void*>(v_ptr)
