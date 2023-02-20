"""DGL Runtime NDArray API.

dgl.ndarray provides a minimum runtime array structure to be
used with C++ library.
"""
# pylint: disable=invalid-name,unused-import
from __future__ import absolute_import as _abs

import ctypes
import functools
import operator

import numpy as _np

from . import backend as F
from ._ffi.function import _init_api
from ._ffi.ndarray import (
    _set_class_ndarray,
    context,
    DGLContext,
    DGLDataType,
    empty,
    empty_shared_mem,
    from_dlpack,
    NDArrayBase,
    numpyasarray,
)
from ._ffi.object import ObjectBase, register_object


class NDArray(NDArrayBase):
    """Lightweight NDArray class for DGL framework."""

    def __len__(self):
        return functools.reduce(operator.mul, self.shape, 1)

    def shared_memory(self, name):
        """Return a copy of the ndarray in shared memory

        Parameters
        ----------
        name : str
            The name of the shared memory

        Returns
        -------
        NDArray
        """
        return empty_shared_mem(name, True, self.shape, self.dtype).copyfrom(
            self
        )


def cpu(dev_id=0):
    """Construct a CPU device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx : DGLContext
        The created context
    """
    return DGLContext(1, dev_id)


def gpu(dev_id=0):
    """Construct a CPU device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx : DGLContext
        The created context
    """
    return DGLContext(2, dev_id)


def array(arr, ctx=cpu(0)):
    """Create an array from source arr.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to be copied from

    ctx : DGLContext, optional
        The device context to create the array

    Returns
    -------
    ret : NDArray
        The created array
    """
    if not isinstance(arr, (_np.ndarray, NDArray)):
        arr = _np.array(arr)
    return empty(arr.shape, arr.dtype, ctx).copyfrom(arr)


def zerocopy_from_numpy(np_data):
    """Create an array that shares the given numpy data.

    Parameters
    ----------
    np_data : numpy.ndarray
        The numpy data

    Returns
    -------
    NDArray
        The array
    """
    arr, _ = numpyasarray(np_data)
    handle = ctypes.pointer(arr)
    return NDArray(handle, is_view=True)


def cast_to_signed(arr):
    """Cast this NDArray from unsigned integer to signed one.

    uint64 -> int64
    uint32 -> int32

    Useful for backends with poor signed integer support (e.g., TensorFlow).

    Parameters
    ----------
    arr : NDArray
        Input array

    Returns
    -------
    NDArray
        Cased array
    """
    return _CAPI_DGLArrayCastToSigned(arr)


def get_shared_mem_array(name, shape, dtype):
    """Get a tensor from shared memory with specific name

    Parameters
    ----------
    name : str
        The unique name of the shared memory
    shape : tuple of int
        The shape of the returned tensor
    dtype : F.dtype
        The dtype of the returned tensor

    Returns
    -------
    F.tensor
        The tensor got from shared memory.
    """
    new_arr = empty_shared_mem(
        name, False, shape, F.reverse_data_type_dict[dtype]
    )
    dlpack = new_arr.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)


def create_shared_mem_array(name, shape, dtype):
    """Create a tensor from shared memory with the specific name

    Parameters
    ----------
    name : str
        The unique name of the shared memory
    shape : tuple of int
        The shape of the returned tensor
    dtype : F.dtype
        The dtype of the returned tensor

    Returns
    -------
    F.tensor
        The created tensor.
    """
    new_arr = empty_shared_mem(
        name, True, shape, F.reverse_data_type_dict[dtype]
    )
    dlpack = new_arr.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)


def exist_shared_mem_array(name):
    """Check the existence of shared-memory array.

    Parameters
    ----------
    name : str
        The name of the shared-memory array.

    Returns
    -------
    bool
        The existence of the array
    """
    return _CAPI_DGLExistSharedMemArray(name)


class SparseFormat:
    """Format code"""

    ANY = 0
    COO = 1
    CSR = 2
    CSC = 3

    FORMAT2STR = {
        0: "ANY",
        1: "COO",
        2: "CSR",
        3: "CSC",
    }


@register_object("aten.SparseMatrix")
class SparseMatrix(ObjectBase):
    """Sparse matrix object class in C++ backend."""

    @property
    def format(self):
        """Sparse format enum

        Returns
        -------
        int
        """
        return _CAPI_DGLSparseMatrixGetFormat(self)

    @property
    def num_rows(self):
        """Number of rows.

        Returns
        -------
        int
        """
        return _CAPI_DGLSparseMatrixGetNumRows(self)

    @property
    def num_cols(self):
        """Number of rows.

        Returns
        -------
        int
        """
        return _CAPI_DGLSparseMatrixGetNumCols(self)

    @property
    def indices(self):
        """Index arrays.

        Returns
        -------
        list of ndarrays
        """
        ret = [_CAPI_DGLSparseMatrixGetIndices(self, i) for i in range(3)]
        return [F.zerocopy_from_dgl_ndarray(arr) for arr in ret]

    @property
    def flags(self):
        """Flag arrays

        Returns
        -------
        list of boolean
        """
        return _CAPI_DGLSparseMatrixGetFlags(self)

    def __getstate__(self):
        return (
            self.format,
            self.num_rows,
            self.num_cols,
            self.indices,
            self.flags,
        )

    def __setstate__(self, state):
        fmt, nrows, ncols, indices, flags = state
        indices = [F.zerocopy_to_dgl_ndarray(idx) for idx in indices]
        self.__init_handle_by_constructor__(
            _CAPI_DGLCreateSparseMatrix, fmt, nrows, ncols, indices, flags
        )

    def __repr__(self):
        return 'SparseMatrix(fmt="{}", shape=({},{}))'.format(
            SparseFormat.FORMAT2STR[self.format], self.num_rows, self.num_cols
        )


_set_class_ndarray(NDArray)
_init_api("dgl.ndarray")
_init_api("dgl.ndarray.uvm", __name__)

# An array representing null (no value) that can be safely converted to
# other backend tensors.
NULL = {
    "int64": array(_np.array([], dtype=_np.int64)),
    "int32": array(_np.array([], dtype=_np.int32)),
}
