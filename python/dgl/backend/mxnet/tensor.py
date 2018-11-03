from __future__ import absolute_import

import numpy as np
import mxnet as mx
import mxnet.ndarray as nd

def data_type_dict():
    return {'float16' : np.float16,
            'float32' : np.float32,
            'float64' : np.float64,
            'uint8'   : np.uint8,
            'int8'    : np.int8,
            'int16'   : np.int16,
            'int32'   : np.int32,
            'int64'   : np.int64}

def cpu():
    return mx.cpu()

def tensor(data, dtype=None):
    return nd.array(data, dtype=dtype)

def coo_matrix(idx, dat, shape):
    # NOTE: use a csr_matrix for the coo input
    return nd.sparse.csr_matrix((data, (idx[0], idx[1])), shape)

def csr_matrix(data, indices, indptr, shape):
    return nd.sparse.csr_matrix((data, indices, indptr), shape)

def is_tensor(obj):
    return isinstance(obj, nd.NDArray)

def shape(input):
    # NOTE: the input cannot be a symbol
    return input.shape

def dtype(input):
    # NOTE: the input cannot be a symbol
    return input.dtype

def context(input):
    return input.context

def astype(input, ty):
    return nd.cast(input, ty)

def asnumpy(input):
    return input.asnumpy()

def copy_to(input, ctx):
    return input.as_in_context(ctx)

def sum(input, dim):
    return nd.sum(input, axis=dim)

def max(input, dim):
    return nd.max(input, axis=dim)

def cat(seq, dim):
    return nd.concat(*seq, dim=dim)

def split(x, sizes_or_sections, dim):
    if isinstance(sizes_or_sections, list):
        # TODO: fallback to numpy is unfortunate
        np_arr = x.asnumpy()
        indices = np.cumsum(sizes_or_sections)[:-1]
        res = np.split(np_arr, indices, axis=dim)
        return [tensor(arr, dtype=x.dtype) for arr in res]
    else:
        return nd.split(x, sizes_or_sections, axis=dim)

def gather_row(data, row_index):
    if isinstance(row_index, nd.NDArray):
        return nd.take(data, row_index)
    else:
        return data[row_index,]

# narrow_row not enabled

def scatter_row(data, row_index, value):
    return mx.nd.contrib.index_copy(data, row_index, value)

def scatter_row_inplace(data, row_index, value):
    data[row_index] = value

def squeeze(input, dim):
    return nd.squeeze(input, axis=dim)

def unsqueeze(input, dim):
    return nd.expand_dims(input, axis=dim)

def reshape(input, shape):
    # NOTE: the input cannot be a symbol
    return nd.reshape(input ,shape)

def zeros(shape, dtype):
    return nd.zeros(shape, dtype=dtype)

def ones(shape, dtype):
    return nd.ones(shape, dtype=dtype)

def spmm(x, y):
    return nd.dot(x, y)

def unique(input):
    # TODO: fallback to numpy is unfortunate
    tmp = input.asnumpy()
    tmp = np.unique(tmp)
    return nd.array(tmp, ctx=input.context, dtype=input.dtype)

def full_1d(length, fill_value):
    return fill_value + nd.zeros((length,))

def nonzero_1d(input):
    # TODO: fallback to numpy is unfortunate
    tmp = input.asnumpy()
    tmp = np.nonzero(tmp)[0]
    return nd.array(tmp, ctx=input.context, dtype=input.dtype)

def sort_1d(input):
    # TODO: this isn't an ideal implementation.
    val = nd.sort(input, is_ascend=True)
    idx = nd.argsort(input, is_ascend=True)
    idx = nd.cast(idx, dtype='int64')
    return val, idx

def arange(start, stop):
    return nd.arange(start, stop, dtype=np.int64)

def zerocopy_to_dlpack(arr):
    return arr.to_dlpack_for_read()

def zerocopy_from_dlpack(dlpack_arr):
    return nd.from_dlpack(dlpack_arr)

def zerocopy_to_numpy(arr):
    # NOTE: not zerocopy
    return arr.asnumpy()

def zerocopy_from_numpy(np_data):
    # NOTE: not zerocopy
    return nd.array(np_data, dtype=np_data.dtype)
