"""DGL Runtime NDArray API.

dgl.ndarray provides a minimum runtime array API to unify
different array libraries used as backend.
"""
# pylint: disable=invalid-name,unused-import
from __future__ import absolute_import as _abs
import numpy as _np

from ._ffi.ndarray import TVMContext, TVMType, NDArrayBase
from ._ffi.ndarray import context, empty, from_dlpack
from ._ffi.ndarray import _set_class_ndarray

class NDArray(NDArrayBase):
    """Lightweight NDArray class for DGL framework."""
    pass

def cpu(dev_id=0):
    """Construct a CPU device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx : TVMContext
        The created context
    """
    return TVMContext(1, dev_id)


def gpu(dev_id=0):
    """Construct a CPU device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx : TVMContext
        The created context
    """
    return TVMContext(2, dev_id)

def array(arr, ctx=cpu(0)):
    """Create an array from source arr.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to be copied from

    ctx : TVMContext, optional
        The device context to create the array

    Returns
    -------
    ret : NDArray
        The created array
    """
    if not isinstance(arr, (_np.ndarray, NDArray)):
        arr = _np.array(arr)
    return empty(arr.shape, arr.dtype, ctx).copyfrom(arr)

_set_class_ndarray(NDArray)
