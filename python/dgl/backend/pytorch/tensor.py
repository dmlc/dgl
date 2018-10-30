from __future__ import absolute_import

import torch as th
from torch.utils import dlpack

def data_type_dict():
    return {'float16' : th.float16,
            'float32' : th.float32,
            'float64' : th.float64,
            'uint8'   : th.uint8,
            'int8'    : th.int8,
            'int16'   : th.int16,
            'int32'   : th.int32,
            'int64'   : th.int64}

def tensor(data, dtype=None):
    return th.tensor(data, dtype)

def coo_tensor(idx, dat, shape):
    return th.sparse.FloatTensor(idx, dat, shape)

def is_tensor(obj):
    return isinstance(obj, th.Tensor)

def shape(input):
    return input.shape

def dtype(input):
    return input.dtype

def context(input):
    return input.device

def astype(input, ty):
    return input.type(ty)

def asnumpy(input):
    return input.cpu().numpy()

def copy_to(input, ctx):
    if ctx.type == 'cpu':
        return input.cpu()
    elif ctx.type == 'cuda':
        th.cuda.set_device(ctx.index)
        return input.cuda()
    else:
        raise RuntimeError('Invalid context', ctx)

def sum(input, dim):
    return th.sum(input, dim=dim)

def max(input, dim):
    return th.max(input, dim=dim)

def cat(seq, dim):
    return th.cat(seq, dim=dim)

def split(input, sizes_or_sections, dim):
    return th.split(input, sizes_or_sections, dim)

def gather_row(data, row_index):
    return th.index_select(data, 0, row_index)

def scatter_row(data, row_index, value):
    return data.index_copy(0, row_index, value)

def scatter_row_inplace(data, row_index, value):
    data[row_index] = value

def squeeze(input, dim):
    return th.squeeze(input, dim)

def unsqueeze(input, dim):
    return th.unsqueeze(input, dim)

def reshape(input, shape):
    return th.reshape(input ,shape)

def zeros(shape, dtype):
    return th.zeros(shape, dtype=dtype)

def ones(shape, dtype):
    return th.ones(shape, dtype=dtype)

def spmm(x, y):
    return th.spmm(x, y)

def unique(input):
    return th.unique(input)

def full_1d(length, fill_value):
    return th.full((length,), fill_value)

def nonzero_1d(input):
    return th.nonzero(input).squeeze()
