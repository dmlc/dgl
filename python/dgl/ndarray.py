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

from ._ffi.object import register_object, ObjectBase
from ._ffi.function import _init_api
from ._ffi.ndarray import DGLContext, DGLType, NDArrayBase
from ._ffi.ndarray import context, empty, from_dlpack, numpyasarray
from ._ffi.ndarray import _set_class_ndarray
from . import backend as F

class NDArray(NDArrayBase):
    """Lightweight NDArray class for DGL framework."""
    def __len__(self):
        return functools.reduce(operator.mul, self.shape, 1)

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

def null():
    """Return a ndarray representing null value. It can be safely converted
    to other backend tensors.

    Returns
    -------
    NDArray
        A null array
    """
    return array(_np.array([], dtype=_np.int64))

class SparseFormat:
    """Format code"""
    ANY = 0
    COO = 1
    CSR = 2
    CSC = 3

    FORMAT2STR = {
        0 : 'ANY',
        1 : 'COO',
        2 : 'CSR',
        3 : 'CSC',
    }

@register_object('aten.SparseMatrix')
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
        #return [F.zerocopy_from_dgl_ndarray(v.data) for v in ret]

    @property
    def flags(self):
        """Flag arrays

        Returns
        -------
        list of boolean
        """
        return [v.data for v in _CAPI_DGLSparseMatrixGetFlags(self)]

    def __getstate__(self):
        return self.format, self.num_rows, self.num_cols, self.indices, self.flags

    def __setstate__(self, state):
        fmt, nrows, ncols, indices, flags = state
        indices = [F.zerocopy_to_dgl_ndarray(idx) for idx in indices]
        self.__init_handle_by_constructor__(
            _CAPI_DGLCreateSparseMatrix, fmt, nrows, ncols, indices, flags)

    def __repr__(self):
        return 'SparseMatrix(fmt="{}", shape=({},{}))'.format(
            SparseFormat.FORMAT2STR[self.format], self.num_rows, self.num_cols)

_set_class_ndarray(NDArray)
_init_api("dgl.ndarray")
