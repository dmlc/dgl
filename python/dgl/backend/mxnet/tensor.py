from __future__ import absolute_import

import numpy as np
import mxnet as mx
import mxnet.ndarray as nd

def data_type_dict():
    return {'float16' : 'float16',
            'float32' : 'float32',
            'float64' : 'float64',
            'uint8'   : 'uint8',
            'int8'    : 'int8',
            'int16'   : 'int16',
            'int32'   : 'int32',
            'int64'   : 'int64'}

def tensor(data, dtype=None):
    return mx.nd.array(data, dtype)

# coo_matrix is not enabled

def csr_matrix(data, indices, indptr, shape):
    return mx.nd.sparse.csr_matrix((data, (indices, indptr)), shape)

def is_tensor(obj):
    return isinstance(obj, mx.nd.NDArray)

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
