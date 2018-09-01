from __future__ import absolute_import

import torch as th
import scipy.sparse
import dgl.context as context

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

def to_context(x, ctx):
    if ctx is None:
        return x
    elif ctx.device == 'gpu':
        th.cuda.set_device(ctx.device_id)
        return x.cuda()
    elif ctx.device == 'cpu':
        return x.cpu()
    else:
        raise RuntimeError('Invalid context', ctx)

def get_context(x):
    if x.device.type == 'cpu':
        return context.cpu()
    else:
        return context.gpu(x.device.index)
