# pylint: disable=invalid-name, unused-import
"""Runtime NDArray api"""
from __future__ import absolute_import

import sys
import ctypes
import numpy as np
from .base import _LIB, check_call, c_array, string_types, _FFI_MODE, c_str
from .runtime_ctypes import DGLType, DGLContext, DGLArray, DGLArrayHandle
from .runtime_ctypes import TypeCode, dgl_shape_index_t


IMPORT_EXCEPT = RuntimeError if _FFI_MODE == "cython" else ImportError

try:
    # pylint: disable=wrong-import-position
    if _FFI_MODE == "ctypes":
        raise ImportError()
    if sys.version_info >= (3, 0):
        from ._cy3.core import _set_class_ndarray, _reg_extension, _make_array, _from_dlpack
        from ._cy3.core import NDArrayBase as _NDArrayBase
    else:
        from ._cy2.core import _set_class_ndarray, _reg_extension, _make_array, _from_dlpack
        from ._cy2.core import NDArrayBase as _NDArrayBase
except IMPORT_EXCEPT:
    # pylint: disable=wrong-import-position
    from ._ctypes.ndarray import _set_class_ndarray, _reg_extension, _make_array, _from_dlpack
    from ._ctypes.ndarray import NDArrayBase as _NDArrayBase

def context(dev_type, dev_id=0):
    """Construct a DGL context with given device type and id.

    Parameters
    ----------
    dev_type: int or str
        The device type mask or name of the device.

    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx: DGLContext
        The corresponding context.

    Examples
    --------
    Context can be used to create reflection of context by
    string representation of the device type.

    .. code-block:: python

      assert dgl.context("cpu", 1) == dgl.cpu(1)
      assert dgl.context("gpu", 0) == dgl.gpu(0)
      assert dgl.context("cuda", 0) == dgl.gpu(0)
    """
    if isinstance(dev_type, string_types):
        dev_type = dev_type.split()[0]
        if dev_type not in DGLContext.STR2MASK:
            raise ValueError("Unknown device type %s" % dev_type)
        dev_type = DGLContext.STR2MASK[dev_type]
    return DGLContext(dev_type, dev_id)


def numpyasarray(np_data):
    """Return a DGLArray representation of a numpy array.
    """
    data = np_data
    assert data.flags['C_CONTIGUOUS']
    arr = DGLArray()
    shape = c_array(dgl_shape_index_t, data.shape)
    arr.data = data.ctypes.data_as(ctypes.c_void_p)
    arr.shape = shape
    arr.strides = None
    arr.dtype = DGLType(np.dtype(data.dtype).name)
    arr.ndim = data.ndim
    # CPU device
    arr.ctx = context(1, 0)
    return arr, shape


def empty(shape, dtype="float32", ctx=context(1, 0)):
    """Create an empty array given shape and device

    Parameters
    ----------
    shape : tuple of int
        The shape of the array

    dtype : type or str
        The data type of the array.

    ctx : DGLContext
        The context of the array

    Returns
    -------
    arr : dgl.nd.NDArray
        The array dgl supported.
    """
    shape = c_array(dgl_shape_index_t, shape)
    ndim = ctypes.c_int(len(shape))
    handle = DGLArrayHandle()
    dtype = DGLType(dtype)
    check_call(_LIB.DGLArrayAlloc(
        shape, ndim,
        ctypes.c_int(dtype.type_code),
        ctypes.c_int(dtype.bits),
        ctypes.c_int(dtype.lanes),
        ctx.device_type,
        ctx.device_id,
        ctypes.byref(handle)))
    return _make_array(handle, False)


def empty_shared_mem(name, is_create, shape, dtype="float32"):
    """Create an empty array with shared memory given shape and dtype

    Parameters
    ----------
    name : string
        The name of the shared memory. It's a file name in Unix.

    is_create : bool
        Whether to create the shared memory or use the one created by somewhere else.

    shape : tuple of int
        The shape of the array

    dtype : type or str
        The data type of the array.

    Returns
    -------
    arr : dgl.nd.NDArray
        The array dgl supported.
    """
    name = ctypes.c_char_p(name.encode('utf-8'))
    shape = c_array(dgl_shape_index_t, shape)
    ndim = ctypes.c_int(len(shape))
    handle = DGLArrayHandle()
    dtype = DGLType(dtype)
    check_call(_LIB.DGLArrayAllocSharedMem(
        name, shape, ndim,
        ctypes.c_int(dtype.type_code),
        ctypes.c_int(dtype.bits),
        ctypes.c_int(dtype.lanes),
        is_create,
        ctypes.byref(handle)))
    return _make_array(handle, False)


def from_dlpack(dltensor):
    """Produce an array from a DLPack tensor without memory copy.
    Retrieves the underlying DLPack tensor's pointer to create an array from the
    data. Removes the original DLPack tensor's destructor as now the array is
    responsible for destruction.

    Parameters
    ----------
    dltensor : DLPack tensor
        Input DLManagedTensor, can only be consumed once.

    Returns
    -------
    arr: dgl.nd.NDArray
        The array view of the tensor data.
    """
    return _from_dlpack(dltensor)


class NDArrayBase(_NDArrayBase):
    """A simple Device/CPU Array object in runtime."""
    @property
    def shape(self):
        """Shape of this array"""
        return tuple(self.handle.contents.shape[i] for i in range(self.handle.contents.ndim))

    @property
    def dtype(self):
        """Type of this array"""
        return str(self.handle.contents.dtype)

    @property
    def ctx(self):
        """context of this array"""
        return self.handle.contents.ctx

    @property
    def context(self):
        """context of this array"""
        return self.ctx

    def __hash__(self):
        return ctypes.cast(self.handle, ctypes.c_void_p).value

    def __eq__(self, other):
        return self.same_as(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def same_as(self, other):
        """Check object identity equality

        Parameters
        ----------
        other : object
            The other object to compare to

        Returns
        -------
        same : bool
            Whether other is same as self.
        """
        if not isinstance(other, NDArrayBase):
            return False
        return self.__hash__() == other.__hash__()

    def __setitem__(self, in_slice, value):
        """Set ndarray value"""
        if (not isinstance(in_slice, slice) or
                in_slice.start is not None
                or in_slice.stop is not None):
            raise ValueError('Array only support set from numpy array')
        if isinstance(value, NDArrayBase):
            if value.handle is not self.handle:
                value.copyto(self)
        elif isinstance(value, (np.ndarray, np.generic)):
            self.copyfrom(value)
        else:
            raise TypeError('type %s not supported' % str(type(value)))

    def copyfrom(self, source_array):
        """Perform a synchronized copy from the array.

        Parameters
        ----------
        source_array : array_like
            The data source we should like to copy from.

        Returns
        -------
        arr : NDArray
            Reference to self.
        """
        if isinstance(source_array, NDArrayBase):
            source_array.copyto(self)
            return self

        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.asarray(source_array, dtype=self.dtype)
            except:
                raise TypeError('array must be an array_like data,' +
                                'type %s is not supported' % str(type(source_array)))
        t = DGLType(self.dtype)
        shape, dtype = self.shape, self.dtype
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t.lanes = 1
            dtype = str(t)

        if source_array.shape != shape:
            raise ValueError("array shape do not match the shape of NDArray {0} vs {1}".format(
                source_array.shape, shape))
        source_array = np.ascontiguousarray(source_array, dtype=dtype)
        assert source_array.flags['C_CONTIGUOUS']
        data = source_array.ctypes.data_as(ctypes.c_void_p)
        nbytes = ctypes.c_size_t(source_array.size * source_array.dtype.itemsize)
        check_call(_LIB.DGLArrayCopyFromBytes(self.handle, data, nbytes))
        return self

    def __repr__(self):
        res = "dgl.{0}@{1}".format(self.asnumpy().__repr__(), self.context)
        return res

    def __str__(self):
        return str(self.asnumpy())

    def asnumpy(self):
        """Convert this array to numpy array

        Returns
        -------
        np_arr : numpy.ndarray
            The corresponding numpy array.
        """
        t = DGLType(self.dtype)
        shape, dtype = self.shape, self.dtype
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t.lanes = 1
            dtype = str(t)
        np_arr = np.empty(shape, dtype=dtype)
        assert np_arr.flags['C_CONTIGUOUS']
        data = np_arr.ctypes.data_as(ctypes.c_void_p)
        nbytes = ctypes.c_size_t(np_arr.size * np_arr.dtype.itemsize)
        check_call(_LIB.DGLArrayCopyToBytes(self.handle, data, nbytes))
        return np_arr

    def copyto(self, target):
        """Copy array to target

        Parameters
        ----------
        target : NDArray
            The target array to be copied, must have same shape as this array.
        """
        if isinstance(target, DGLContext):
            target = empty(self.shape, self.dtype, target)
        if isinstance(target, NDArrayBase):
            check_call(_LIB.DGLArrayCopyFromTo(
                self.handle, target.handle, None))
        else:
            raise ValueError("Unsupported target type %s" % str(type(target)))
        return target


def free_extension_handle(handle, type_code):
    """Free c++ extension type handle

    Parameters
    ----------
    handle : ctypes.c_void_p
        The handle to the extension type.

    type_code : int
         The tyoe code
    """
    check_call(_LIB.DGLExtTypeFree(handle, ctypes.c_int(type_code)))

def register_extension(cls, fcreate=None):
    """Register a extension class to DGL.

    After the class is registered, the class will be able
    to directly pass as Function argument generated by DGL.

    Parameters
    ----------
    cls : class
        The class object to be registered as extension.

    Note
    ----
    The registered class is requires one property: _dgl_handle and a class attribute _dgl_tcode.

    - ```_dgl_handle``` returns integer represents the address of the handle.
    - ```_dgl_tcode``` gives integer represents type code of the class.

    Returns
    -------
    cls : class
        The class being registered.

    fcreate : function, optional
        The creation function to create a class object given handle value.

    Example
    -------
    The following code registers user defined class
    MyTensor to be DLTensor compatible.

    .. code-block:: python

       @dgl.register_extension
       class MyTensor(object):
           _dgl_tcode = dgl.TypeCode.ARRAY_HANDLE

           def __init__(self):
               self.handle = _LIB.NewDLTensor()

           @property
           def _dgl_handle(self):
               return self.handle.value
    """
    if fcreate and cls._dgl_tcode < TypeCode.EXT_BEGIN:
        raise ValueError("Cannot register create when extension tcode is same as buildin")
    _reg_extension(cls, fcreate)
    return cls
