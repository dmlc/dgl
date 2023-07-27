from ..runtime_ctypes import DGLArrayHandle as PyDGLArrayHandle
from cpython cimport PyCapsule_Destructor

cdef const char* _c_str_dltensor = "dltensor"
cdef const char* _c_str_used_dltensor = "used_dltensor"


cdef _c_dlpack_deleter(object pycaps):
    cdef DLManagedTensor* dltensor
    if pycapsule.PyCapsule_IsValid(pycaps, _c_str_dltensor):
        dltensor = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(pycaps, _c_str_dltensor)
        DGLDLManagedTensorCallDeleter(dltensor)


def _from_dlpack(object dltensor):
    cdef DLManagedTensor* ptr
    cdef DGLArrayHandle chandle
    if pycapsule.PyCapsule_IsValid(dltensor, _c_str_dltensor):
        ptr = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(dltensor, _c_str_dltensor)
        CALL(DGLArrayFromDLPack(ptr, &chandle))
        # set name and destructor to be empty
        pycapsule.PyCapsule_SetDestructor(dltensor, NULL)
        pycapsule.PyCapsule_SetName(dltensor, _c_str_used_dltensor)
        return c_make_array(chandle, 0)
    raise ValueError("Expect a dltensor field, pycapsule.PyCapsule can only be consumed once")


cdef class NDArrayBase:
    cdef DGLArray* chandle
    cdef int c_is_view

    cdef inline _set_handle(self, handle):
        cdef unsigned long long ptr
        if handle is None:
            self.chandle = NULL
        else:
            ptr = ctypes.cast(handle, ctypes.c_void_p).value
            self.chandle = <DGLArray*>(ptr)

    property _dgl_handle:
        def __get__(self):
            return <unsigned long long>self.chandle

    property handle:
        def __get__(self):
            if self.chandle == NULL:
                return None
            else:
                return ctypes.cast(
                    <unsigned long long>self.chandle, PyDGLArrayHandle)

        def __set__(self, value):
            self._set_handle(value)

    def __init__(self, handle, is_view):
        self._set_handle(handle)
        self.c_is_view = is_view

    def __dealloc__(self):
        if self.c_is_view == 0:
            CALL(DGLArrayFree(self.chandle))

    def to_dlpack(self, alignment=0):
        """Produce an array from a DLPack Tensor without copying memory

        Args
        -------
        alignment: int, default to be 0
        Indicates the alignment requirement when converting to dlpack. Will copy to a 
        new tensor if the alignment requirement is not satisfied. 
        0 means no alignment requirement.
        
        Returns
        -------
        dlpack : DLPack tensor view of the array data
        """
        cdef DLManagedTensor* dltensor
        if self.c_is_view != 0:
            raise ValueError("to_dlpack do not work with memory views")
        CALL(DGLArrayToDLPack(self.chandle, &dltensor, alignment))
        return pycapsule.PyCapsule_New(dltensor, _c_str_dltensor, <PyCapsule_Destructor>_c_dlpack_deleter)


cdef c_make_array(void* chandle, is_view):
    ret = _CLASS_NDARRAY(None, is_view)
    (<NDArrayBase>ret).chandle = <DGLArray*>chandle
    return ret


cdef _DGL_COMPATS = ()

cdef _DGL_EXT_RET = {}

def _reg_extension(cls, fcreate):
    global _DGL_COMPATS
    _DGL_COMPATS += (cls,)
    if fcreate:
        _DGL_EXT_RET[cls._dgl_tcode] = fcreate


def _make_array(handle, is_view):
    cdef unsigned long long ptr
    ptr = ctypes.cast(handle, ctypes.c_void_p).value
    return c_make_array(<void*>ptr, is_view)

cdef object _CLASS_NDARRAY = None

def _set_class_ndarray(cls):
    global _CLASS_NDARRAY
    _CLASS_NDARRAY = cls
