from __future__ import absolute_import

import torch as th

from .._ffi.runtime_ctypes import TVMType, TVMContext, TVMArray
from .._ffi.runtime_ctypes import TypeCode, tvm_shape_index_t

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

def astype(a, ty):
    return a.type(ty)

def asnumpy(a):
    return a.cpu().numpy()

def pack(tensors):
    return th.cat(tensors)

def unpack(x, indices_or_sections=1):
    return th.split(x, indices_or_sections)

def shape(x):
    return x.shape

def isinteger(x):
    return x.dtype in [th.int, th.int8, th.int16, th.int32, th.int64]

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

def get_context(arr):
    if arr.device.type == 'cpu':
        return TVMContext(TVMContext.STR2MASK['cpu'], 0)
    else:
        return TVMContext(
                TVMContext.STR2MASK[arr.device.type], arr.device.index)

def asdglarray(arr):
    assert arr.is_contiguous()
    rst = TVMArray()
    rst.data = arr.data_ptr()
    rst.shape = c_array(tvm_shape_index_t, arr.shape)
    rst.strides = None
    # TODO: dtype
    rst.dtype = TVMType(arr.dtype)
    rst.ndim = arr.ndimension()
    # TODO: ctx
    return rst
