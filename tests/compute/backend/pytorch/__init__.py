from __future__ import absolute_import

import torch as th

def cuda():
    return th.device('cuda')

def equal(a, b):
    return th.equal(a, b)

def allclose(a, b):
    return th.allclose(a, b)

def randn(shape):
    return th.randn(*shape)

def attach_grad(x):
    return x.requires_grad_()

def backward(x, head_gradient=None):
    x.backward(head_gradient)

def grad(x):
    return x.grad

def full(shape, fill_value, dtype, ctx):
    return th.full(shape, fill_value, dtype=dtype, device=ctx)


class record_grad(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

no_grad = th.no_grad
