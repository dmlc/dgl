from ... import _api_internal
from ..base import string_types
from ..object_generic import _set_class_object_base

"""Maps object type to its constructor"""
OBJECT_TYPE = []

def _register_object(int index, object cls):
    """register object class"""
    while len(OBJECT_TYPE) <= index:
        OBJECT_TYPE.append(None)
    OBJECT_TYPE[index] = cls


cdef inline object make_ret_object(void* chandle):
    global OBJECT_TYPE
    cdef int tindex
    cdef list object_type
    cdef object cls
    object_type = OBJECT_TYPE
    CALL(DGLObjectGetTypeIndex(chandle, &tindex))
    if tindex < len(object_type):
        cls = object_type[tindex]
        if cls is not None:
            obj = cls.__new__(cls)
        else:
            obj = ObjectBase.__new__(ObjectBase)
    else:
        obj = ObjectBase.__new__(ObjectBase)
    (<ObjectBase>obj).chandle = chandle
    return obj


cdef class ObjectBase:
    cdef void* chandle

    cdef _set_handle(self, handle):
        cdef unsigned long long ptr
        if handle is None:
            self.chandle = NULL
        else:
            ptr = handle.value
            self.chandle = <void*>(ptr)

    property handle:
        def __get__(self):
            if self.chandle == NULL:
                return None
            else:
                return ctypes_handle(self.chandle)

        def __set__(self, value):
            self._set_handle(value)

    def __dealloc__(self):
        CALL(DGLObjectFree(self.chandle))

    def __getattr__(self, name):
        cdef DGLValue ret_val
        cdef int ret_type_code, ret_succ
        CALL(DGLObjectGetAttr(self.chandle, c_str(name),
                            &ret_val, &ret_type_code, &ret_succ))
        if ret_succ == 0:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (type(self), name))
        return make_ret(ret_val, ret_type_code)

    def __init_handle_by_constructor__(self, fconstructor, *args):
        """Initialize the handle by calling constructor function.

        Parameters
        ----------
        fconstructor : Function
            Constructor function.

        args: list of objects
            The arguments to the constructor

        Note
        ----
        We have a special calling convention to call constructor functions.
        So the return handle is directly set into the Object object
        instead of creating a new Object.
        """
        cdef void* chandle
        ConstructorCall(
            (<FunctionBase>fconstructor).chandle,
            kObjectHandle, args, &chandle)
        self.chandle = chandle

_set_class_object_base(ObjectBase)
