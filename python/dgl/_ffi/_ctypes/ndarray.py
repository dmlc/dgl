# pylint: disable=invalid-name
"""Runtime NDArray api"""
from __future__ import absolute_import

import ctypes

from ..base import _LIB, c_str, check_call
from ..runtime_ctypes import DGLArrayHandle
from .types import (
    _return_handle,
    _wrap_arg_func,
    C_TO_PY_ARG_SWITCH,
    RETURN_SWITCH,
)

DGLPyCapsuleDestructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_c_str_dltensor = c_str("dltensor")
_c_str_used_dltensor = c_str("used_dltensor")


# used for PyCapsule manipulation
if hasattr(ctypes, "pythonapi"):
    ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object


def _from_dlpack(dltensor):
    dltensor = ctypes.py_object(dltensor)
    if ctypes.pythonapi.PyCapsule_IsValid(dltensor, _c_str_dltensor):
        ptr = ctypes.pythonapi.PyCapsule_GetPointer(dltensor, _c_str_dltensor)
        # XXX(minjie): The below cast should be unnecessary given the code to
        #   set restype of PyCapsule calls. But weirdly, this does not
        #   work out always.
        ptr = ctypes.cast(ptr, ctypes.c_void_p)
        handle = DGLArrayHandle()
        check_call(_LIB.DGLArrayFromDLPack(ptr, ctypes.byref(handle)))
        ctypes.pythonapi.PyCapsule_SetName(dltensor, _c_str_used_dltensor)
        ctypes.pythonapi.PyCapsule_SetDestructor(
            dltensor, DGLPyCapsuleDestructor(0)
        )
        return _make_array(handle, False)
    raise ValueError(
        "Expect a dltensor field, PyCapsule can only be consumed once"
    )


def _dlpack_deleter(pycapsule):
    pycapsule = ctypes.cast(pycapsule, ctypes.py_object)
    if ctypes.pythonapi.PyCapsule_IsValid(pycapsule, _c_str_dltensor):
        ptr = ctypes.pythonapi.PyCapsule_GetPointer(pycapsule, _c_str_dltensor)
        # XXX(minjie): The below cast should be unnecessary given the code to
        #   set restype of PyCapsule calls. But weirdly, this does not
        #   work out always.
        ptr = ctypes.cast(ptr, ctypes.c_void_p)
        _LIB.DGLDLManagedTensorCallDeleter(ptr)
        ctypes.pythonapi.PyCapsule_SetDestructor(
            pycapsule, DGLPyCapsuleDestructor(0)
        )


_c_dlpack_deleter = DGLPyCapsuleDestructor(_dlpack_deleter)


class NDArrayBase(object):
    """A simple Device/CPU Array object in runtime."""

    __slots__ = ["handle", "is_view"]
    # pylint: disable=no-member
    def __init__(self, handle, is_view=False):
        """Initialize the function with handle

        Parameters
        ----------
        handle : DGLArrayHandle
            the handle to the underlying C++ DGLArray
        """
        self.handle = handle
        self.is_view = is_view

    def __del__(self):
        if not self.is_view and _LIB:
            check_call(_LIB.DGLArrayFree(self.handle))

    @property
    def _dgl_handle(self):
        return ctypes.cast(self.handle, ctypes.c_void_p).value

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
        ptr = ctypes.c_void_p()
        check_call(
            _LIB.DGLArrayToDLPack(self.handle, ctypes.byref(ptr), alignment)
        )
        return ctypes.pythonapi.PyCapsule_New(
            ptr, _c_str_dltensor, _c_dlpack_deleter
        )


def _make_array(handle, is_view):
    handle = ctypes.cast(handle, DGLArrayHandle)
    return _CLASS_NDARRAY(handle, is_view)


_DGL_COMPATS = ()


def _reg_extension(cls, fcreate):
    global _DGL_COMPATS
    _DGL_COMPATS += (cls,)
    if fcreate:
        fret = lambda x: fcreate(_return_handle(x))
        RETURN_SWITCH[cls._dgl_tcode] = fret
        C_TO_PY_ARG_SWITCH[cls._dgl_tcode] = _wrap_arg_func(
            fret, cls._dgl_tcode
        )


_CLASS_NDARRAY = None


def _set_class_ndarray(cls):
    global _CLASS_NDARRAY
    _CLASS_NDARRAY = cls
