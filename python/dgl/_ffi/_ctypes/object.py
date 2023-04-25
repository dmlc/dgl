"""ctypes object API."""
from __future__ import absolute_import

import ctypes

from ..base import _LIB, c_str, check_call
from ..object_generic import _set_class_object_base
from .types import (
    _wrap_arg_func,
    C_TO_PY_ARG_SWITCH,
    DGLValue,
    RETURN_SWITCH,
    TypeCode,
)

ObjectHandle = ctypes.c_void_p
__init_by_constructor__ = None

"""Maps object type to its constructor"""
OBJECT_TYPE = {}


def _register_object(index, cls):
    """register object class in python"""
    OBJECT_TYPE[index] = cls


def _return_object(x):
    """Construct a object object from the given DGLValue object"""
    handle = x.v_handle
    if not isinstance(handle, ObjectHandle):
        handle = ObjectHandle(handle)
    tindex = ctypes.c_int()
    check_call(_LIB.DGLObjectGetTypeIndex(handle, ctypes.byref(tindex)))
    cls = OBJECT_TYPE.get(tindex.value, ObjectBase)
    # Avoid calling __init__ of cls, instead directly call __new__
    # This allows child class to implement their own __init__
    obj = cls.__new__(cls)
    obj.handle = handle
    return obj


RETURN_SWITCH[TypeCode.OBJECT_HANDLE] = _return_object
C_TO_PY_ARG_SWITCH[TypeCode.OBJECT_HANDLE] = _wrap_arg_func(
    _return_object, TypeCode.OBJECT_HANDLE
)


class ObjectBase(object):
    """Object base class"""

    __slots__ = ["handle"]

    # pylint: disable=no-member
    def __del__(self):
        if _LIB is not None and hasattr(self, "handle"):
            check_call(_LIB.DGLObjectFree(self.handle))

    def __getattr__(self, name):
        if name == "handle":
            raise AttributeError(
                "'handle' is a reserved attribute name that should not be used"
            )
        ret_val = DGLValue()
        ret_type_code = ctypes.c_int()
        ret_success = ctypes.c_int()
        check_call(
            _LIB.DGLObjectGetAttr(
                self.handle,
                c_str(name),
                ctypes.byref(ret_val),
                ctypes.byref(ret_type_code),
                ctypes.byref(ret_success),
            )
        )
        if not ret_success.value:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (str(type(self)), name)
            )
        return RETURN_SWITCH[ret_type_code.value](ret_val)

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
        # assign handle first to avoid error raising
        self.handle = None
        handle = __init_by_constructor__(
            fconstructor, args
        )  # pylint: disable=not-callable
        if not isinstance(handle, ObjectHandle):
            handle = ObjectHandle(handle)
        self.handle = handle


_set_class_object_base(ObjectBase)
