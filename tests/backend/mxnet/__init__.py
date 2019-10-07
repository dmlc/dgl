from __future__ import absolute_import

import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.autograd as autograd

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

def attach_grad(x):
    x.attach_grad()
    return x

def backward(x, head_gradient=None):
    x.backward(head_gradient)

def grad(x):
    return x.grad

def is_no_grad(x):
    return (x != 0).sum() == 0

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

def sum(x, dim):
    return x.sum(dim)

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

record_grad = autograd.record


class no_grad(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
