from ..runtime_ctypes import TVMArrayHandle

cdef const char* _c_str_dltensor = "dltensor"
cdef const char* _c_str_used_dltensor = "used_dltensor"


cdef void _c_dlpack_deleter(object pycaps):
    cdef DLManagedTensor* dltensor
    if pycapsule.PyCapsule_IsValid(pycaps, _c_str_dltensor):
        dltensor = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(pycaps, _c_str_dltensor)
        TVMDLManagedTensorCallDeleter(dltensor)


def _from_dlpack(object dltensor):
    cdef DLManagedTensor* ptr
    cdef DLTensorHandle chandle
    if pycapsule.PyCapsule_IsValid(dltensor, _c_str_dltensor):
        ptr = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(dltensor, _c_str_dltensor)
        CALL(TVMArrayFromDLPack(ptr, &chandle))
        # set name and destructor to be empty
        pycapsule.PyCapsule_SetDestructor(dltensor, NULL)
        pycapsule.PyCapsule_SetName(dltensor, _c_str_used_dltensor)
        return c_make_array(chandle, 0)
    raise ValueError("Expect a dltensor field, pycapsule.PyCapsule can only be consumed once")


cdef class NDArrayBase:
    cdef DLTensor* chandle
    cdef int c_is_view

    cdef inline _set_handle(self, handle):
        cdef unsigned long long ptr
        if handle is None:
            self.chandle = NULL
        else:
            ptr = ctypes.cast(handle, ctypes.c_void_p).value
            self.chandle = <DLTensor*>(ptr)

    property _tvm_handle:
        def __get__(self):
            return <unsigned long long>self.chandle

    property handle:
        def __get__(self):
            if self.chandle == NULL:
                return None
            else:
                return ctypes.cast(
                    <unsigned long long>self.chandle, TVMArrayHandle)

        def __set__(self, value):
            self._set_handle(value)

    def __init__(self, handle, is_view):
        self._set_handle(handle)
        self.c_is_view = is_view

    def __dealloc__(self):
        if self.c_is_view == 0:
            CALL(TVMArrayFree(self.chandle))

    def to_dlpack(self):
        """Produce an array from a DLPack Tensor without copying memory

        Returns
        -------
        dlpack : DLPack tensor view of the array data
        """
        cdef DLManagedTensor* dltensor
        if self.c_is_view != 0:
            raise ValueError("to_dlpack do not work with memory views")
        CALL(TVMArrayToDLPack(self.chandle, &dltensor))
        return pycapsule.PyCapsule_New(dltensor, _c_str_dltensor, _c_dlpack_deleter)


cdef c_make_array(void* chandle, is_view):
    ret = _CLASS_NDARRAY(None, is_view)
    (<NDArrayBase>ret).chandle = <DLTensor*>chandle
    return ret


cdef _TVM_COMPATS = ()

cdef _TVM_EXT_RET = {}

def _reg_extension(cls, fcreate):
    global _TVM_COMPATS
    _TVM_COMPATS += (cls,)
    if fcreate:
        _TVM_EXT_RET[cls._tvm_tcode] = fcreate


def _make_array(handle, is_view):
    cdef unsigned long long ptr
    ptr = ctypes.cast(handle, ctypes.c_void_p).value
    return c_make_array(<void*>ptr, is_view)

cdef object _CLASS_NDARRAY = None

def _set_class_ndarray(cls):
    global _CLASS_NDARRAY
    _CLASS_NDARRAY = cls
