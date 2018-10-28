from __future__ import absolute_import

import ctypes
import scipy as sp
import torch as th
from torch.utils import dlpack

from .._ffi.base import _LIB, check_call, c_array
from .._ffi.runtime_ctypes import TVMType, TVMContext, TVMArray
from .._ffi.runtime_ctypes import TypeCode, tvm_shape_index_t
from .. import ndarray as nd

# Tensor types
Tensor = th.Tensor
SparseTensor = th.sparse.FloatTensor

# Data types
float16 = th.float16
float32 = th.float32
float64 = th.float64
uint8 = th.uint8
int8 = th.int8
int16 = th.int16
int32 = th.int32
int64 = th.int64

# Operators
tensor = th.tensor
sparse_tensor = th.sparse.FloatTensor
sum = th.sum
max = th.max
stack = th.stack

def astype(a, ty):
    return a.type(ty)

def asnumpy(a):
    return a.cpu().numpy()

def from_numpy(np_data):
    return th.from_numpy(np_data)

def pack(tensors, dim=0):
    return th.cat(tensors, dim)

def unpack(x, indices_or_sections=1):
    return th.split(x, indices_or_sections)

def shape(x):
    return x.shape

def dtype(x):
    return x.dtype

unique = th.unique

def gather_row(data, row_index):
    return th.index_select(data, 0, row_index)

def scatter_row(data, row_index, value):
    return data.index_copy(0, row_index, value)

def broadcast_to(x, to_array):
    return x + th.zeros_like(to_array)

nonzero = th.nonzero
squeeze = th.squeeze
unsqueeze = th.unsqueeze
reshape = th.reshape
zeros = th.zeros
ones = th.ones
zeros = th.zeros
spmm = th.spmm
sort = th.sort
arange = th.arange
mul = th.mul

def to_context(arr, ctx):
    if ctx is None:
        return arr
    elif ctx.device_type == TVMContext.STR2MASK['cuda']:
        th.cuda.set_device(ctx.device_id)
        return arr.cuda()
    elif ctx.device_type == TVMContext.STR2MASK['cpu']:
        return arr.cpu()
    else:
        raise RuntimeError('Invalid context', ctx)

def _get_context(type, index):
    if type == 'cpu':
        return TVMContext(TVMContext.STR2MASK['cpu'], 0)
    else:
        return TVMContext(TVMContext.STR2MASK[type], index)

def get_context(arr):
    return _get_context(arr.device.type, arr.device.index)

_tvmtypes = {
        th.float16: TVMType('float16'),
        th.float32: TVMType('float32'),
        th.float64: TVMType('float64'),
        th.int8: TVMType('int8'),
        th.uint8: TVMType('uint8'),
        th.int16: TVMType('int16'),
        th.int32: TVMType('int32'),
        th.int64: TVMType('int64'),
        }

def get_tvmtype(arr):
    arr_dtype = arr.dtype
    if arr_dtype not in _tvmtypes:
        raise RuntimeError('Unsupported data type:', arr_dtype)
    return _tvmtypes[arr_dtype]

def zerocopy_to_dlpack(arr):
    """Return a dlpack compatible array using zero copy."""
    return dlpack.to_dlpack(arr)

def zerocopy_from_dlpack(dlpack_arr):
    """Return a tensor using zero copy."""
    return dlpack.from_dlpack(dlpack_arr)

def zerocopy_to_numpy(arr):
    """Return a numpy array that shares the data."""
    return arr.numpy()

def zerocopy_from_numpy(np_data):
    """Return a tensor that shares the numpy data."""
    return th.from_numpy(np_data)

def nonzero_1d(arr):
    """Return a 1D tensor with nonzero element indices in a 1D vector"""
    assert arr.dim() == 1
    return th.nonzero(arr)[:, 0]
