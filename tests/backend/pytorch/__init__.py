from __future__ import absolute_import

import torch as th


def cuda():
    return th.device("cuda:0")


def is_cuda_available():
    return th.cuda.is_available()


def array_equal(a, b):
    return th.equal(a.cpu(), b.cpu())


def allclose(a, b, rtol=1e-4, atol=1e-4):
    return th.allclose(a.float().cpu(), b.float().cpu(), rtol=rtol, atol=atol)


def randn(shape):
    return th.randn(*shape)


def full(shape, fill_value, dtype, ctx):
    return th.full(shape, fill_value, dtype=dtype, device=ctx)


def narrow_row_set(x, start, stop, new):
    x[start:stop] = new


def sparse_to_numpy(x):
    return x.to_dense().numpy()


def clone(x):
    return x.clone()


def reduce_sum(x):
    return x.sum()


def softmax(x, dim):
    return th.softmax(x, dim)


def spmm(x, y):
    return th.spmm(x, y)


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
    return x.max(dim)[0]


def min(x, dim):
    return x.min(dim)[0]


def prod(x, dim):
    return x.prod(dim)


def matmul(a, b):
    return a @ b


def dot(a, b):
    return sum(mul(a, b), dim=-1)


def abs(a):
    return a.abs()


def seed(a):
    return th.manual_seed(a)
