from __future__ import absolute_import

import torch as th

def cuda():
    return th.device('cuda:0')

def is_cuda_available():
    return th.cuda.is_available()

def array_equal(a, b):
    return th.equal(a.cpu(), b.cpu())

def allclose(a, b, rtol=1e-4, atol=1e-4):
    return th.allclose(a.float().cpu(),
            b.float().cpu(), rtol=rtol, atol=atol)

def randn(shape):
    return th.randn(*shape)

def attach_grad(x):
    if x.grad is not None:
        x.grad.zero_()
        return x
    else:
        return x.requires_grad_()

def backward(x, head_gradient=None):
    if head_gradient is not None and head_gradient.shape[0] == 1 and len(head_gradient.shape) == 1:
        # Fix for torch 1.3.1
        head_gradient = th.tensor(head_gradient.item()).to(head_gradient.device)
    x.backward(head_gradient)

def grad(x):
    return x.grad

def is_no_grad(x):
    return x.grad is None or (x.grad == 0).all()

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

def sum(x, dim):
    return x.sum(dim)

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

class record_grad(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

no_grad = th.no_grad
