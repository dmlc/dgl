from __future__ import absolute_import

import mxnet as mx
import mxnet.ndarray as nd
import numpy as np


def cuda():
    return mx.gpu()


def is_cuda_available():
    # TODO: Does MXNet have a convenient function to test GPU availability/compilation?
    try:
        a = nd.array([1, 2, 3], ctx=mx.gpu())
        return True
    except mx.MXNetError:
        return False


def array_equal(a, b):
    return nd.equal(a, b).asnumpy().all()


def allclose(a, b, rtol=1e-4, atol=1e-4):
    return np.allclose(a.asnumpy(), b.asnumpy(), rtol=rtol, atol=atol)


def randn(shape):
    return nd.random.randn(*shape)


def full(shape, fill_value, dtype, ctx):
    return nd.full(shape, fill_value, dtype=dtype, ctx=ctx)


def narrow_row_set(x, start, stop, new):
    x[start:stop] = new


def sparse_to_numpy(x):
    return x.asscipy().todense().A


def clone(x):
    return x.copy()


def reduce_sum(x):
    return x.sum()


def softmax(x, dim):
    return nd.softmax(x, axis=dim)


def spmm(x, y):
    return nd.dot(x, y)


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    return a / b


def sum(x, dim, keepdims=False):
    return x.sum(dim, keepdims=keepdims)


def max(x, dim):
    return x.max(dim)


def min(x, dim):
    return x.min(dim)


def prod(x, dim):
    return x.prod(dim)


def matmul(a, b):
    return nd.dot(a, b)


def dot(a, b):
    return nd.sum(mul(a, b), axis=-1)


def abs(a):
    return nd.abs(a)


def seed(a):
    return mx.random.seed(a)
