from __future__ import absolute_import

import torch as th
import scipy.sparse

# Tensor types
Tensor = th.Tensor
SparseTensor = scipy.sparse.spmatrix

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
sum = th.sum
max = th.max

def asnumpy(a):
    return a.cpu().numpy()

def concatenate(tensors, axis=0):
    return th.concatenate(tensors, axis)

def packable(tensors):
    return all(isinstance(x, th.Tensor) and \
               x.dtype == tensors[0].dtype and \
               x.shape[1:] == tensors[0].shape[1:] for x in tensors)

def pack(tensors):
    return th.cat(tensors)

def unpackable(x):
    return isinstance(x, th.Tensor) and x.numel() > 0

def unpack(x):
    return th.split(x, 1)

def shape(x):
    return x.shape

def expand_dims(x, axis):
    return x.unsqueeze(axis)

def prod(x, axis=None, keepdims=None):
    args = ([axis] if axis else []) + ([keepdims] if keepdims else []) 
    return th.prod(x, *args)

def item(x):
    return x.item()

def isinteger(x):
    return x.dtype in [th.int, th.int8, th.int16, th.int32, th.int64]

def isin(x, y):
    assert x.device == y.device
    assert x.dtype == y.dtype
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    return (x[None, :] == y[:, None]).any(-1)

def dtype(x):
    return x.dtype

def astype(x, dtype):
    return x.type(dtype)

ones = th.ones
unique = th.unique

def gather_row(data, row_index):
    return th.index_select(data, 0, row_index)

def scatter_row(data, row_index, value):
    return data.index_copy(0, row_index, value)

def broadcast_to(x, to_array):
    return x + th.zeros_like(to_array)
