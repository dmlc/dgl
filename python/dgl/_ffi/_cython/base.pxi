from ..base import DGLError
from libcpp.vector cimport vector
from libcpp cimport bool
from cpython.version cimport PY_MAJOR_VERSION
from cpython cimport pycapsule
from libc.stdint cimport int64_t, uint64_t, uint8_t, uint16_t
import ctypes

cdef enum DGLTypeCode:
    kInt = 0
    kUInt = 1
    kFloat = 2
    kHandle = 3
    kNull = 4
    kDGLType = 5
    kDGLContext = 6
    kArrayHandle = 7
    kObjectHandle = 8
    kModuleHandle = 9
    kFuncHandle = 10
    kStr = 11
    kBytes = 12
    kNDArrayContainer = 13
    kExtBegin = 15

cdef extern from "dgl/runtime/c_runtime_api.h":
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

    ctypedef struct DGLValue:
        int64_t v_int64
        double v_float64
        void* v_handle
        const char* v_str
        DLDataType v_type
        DLContext v_ctx

ctypedef int64_t dgl_index_t
ctypedef DLTensor* DLTensorHandle
ctypedef DLTensor DGLArray
ctypedef DGLArray* CDGLArrayHandle
ctypedef void* DGLStreamHandle
ctypedef void* DGLRetValueHandle
ctypedef void* DGLFunctionHandle
ctypedef void* ObjectHandle

ctypedef int (*DGLPackedCFunc)(
    DGLValue* args,
    int* type_codes,
    int num_args,
    DGLRetValueHandle ret,
    void* resource_handle)

ctypedef void (*DGLPackedCFuncFinalizer)(void* resource_handle)

cdef extern from "dgl/runtime/c_runtime_api.h":
    void DGLAPISetLastError(const char* msg)
    const char *DGLGetLastError()
    int DGLFuncCall(DGLFunctionHandle func,
                    DGLValue* arg_values,
                    int* type_codes,
                    int num_args,
                    DGLValue* ret_val,
                    int* ret_type_code) nogil
    int DGLFuncFree(DGLFunctionHandle func)
    int DGLCFuncSetReturn(DGLRetValueHandle ret,
                          DGLValue* value,
                          int* type_code,
                          int num_ret)
    int DGLFuncCreateFromCFunc(DGLPackedCFunc func,
                               void* resource_handle,
                               DGLPackedCFuncFinalizer fin,
                               DGLFunctionHandle *out)
    int DGLCbArgToReturn(DGLValue* value, int code)
    int DGLArrayAlloc(dgl_index_t* shape,
                      dgl_index_t ndim,
                      DLDataType dtype,
                      DLContext ctx,
                      DLTensorHandle* out)
    int DGLArrayAllocSharedMem(const char *mem_name,
                               const dgl_index_t *shape,
                               int ndim,
                               int dtype_code,
                               int dtype_bits,
                               int dtype_lanes,
                               bool is_create,
                               CDGLArrayHandle* out)
    int DGLArrayFree(DLTensorHandle handle)
    int DGLArrayCopyFromTo(DLTensorHandle src,
                           DLTensorHandle to,
                           DGLStreamHandle stream)
    int DGLArrayFromDLPack(DLManagedTensor* arr_from,
                           DLTensorHandle* out)
    int DGLArrayToDLPack(DLTensorHandle arr_from,
                         DLManagedTensor** out,
                         int alignment)
    void DGLDLManagedTensorCallDeleter(DLManagedTensor* dltensor)

cdef extern from "dgl/runtime/c_object_api.h":
    int DGLObjectFree(ObjectHandle handle)
    int DGLObjectTypeKey2Index(const char* type_key,
                               int* out_index)
    int DGLObjectGetTypeIndex(ObjectHandle handle,
                              int* out_index)
    int DGLObjectGetAttr(ObjectHandle handle,
                         const char* key,
                         DGLValue* out_value,
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
        raise DGLError(py_str(DGLGetLastError()))


cdef inline object ctypes_handle(void* chandle):
    """Cast C handle to ctypes handle."""
    return ctypes.cast(<unsigned long long>chandle, ctypes.c_void_p)


cdef inline void* c_handle(object handle):
    """Cast C types handle to c handle."""
    cdef unsigned long long v_ptr
    v_ptr = handle.value
    return <void*>(v_ptr)
