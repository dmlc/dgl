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

def packable(tensors):
    return all(isinstance(x, th.Tensor) and \
               x.dtype == tensors[0].dtype and \
               x.shape[1:] == tensors[0].shape[1:] for x in tensors)

def pack(tensors):
    return th.cat(tensors)

def unpack(x):
    return th.split(x, 1)

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
def eq_scalar(x, val):
    return th.eq(x, float(val))
squeeze = th.squeeze
reshape = th.reshape
