import chainer
import chainer.functions as F
import operator
try:
    import cupy
except ImportError:
    cupy = None
from dgl.backend.chainer.utils import *
import numpy as np

def cuda():
    return '@cupy:0'

def is_cuda_available():
    return chainer.cuda.available

def array_equal(a, b):
    return (a.data == b.data).all().item()

def allclose(a, b, rtol=1e-4, atol=1e-4):
    return get_array_module(a, b).allclose(a.data, b.data, rtol, atol)

def randn(shape):
    return chainer.as_variable(np.random.randn(*shape).astype('float32'))

def attach_grad(x):
    if x.grad is not None:
        x.cleargrad()
    else:
        x = chainer.Variable(x.data)    # creates a new variable that returns gradients
    return x

def backward(x, head_gradient=None):
    assert head_gradient is None, "Chainer does not support head gradients"
    x.backward()

def grad(x):
    return x.grad_var

def is_no_grad(x):
    return x.grad is None or (x.grad == 0).all().item()

def full(shape, fill_value, dtype, ctx):
    return chainer.as_variable(
        get_context_module(ctx).full(shape, fill_value, dtype=dtype))

def index_set(x, i, new):
    x.data[i] = new

def narrow_row_set(x, start, stop, new):
    if isinstance(new, chainer.Variable):
        x.data[start:stop] = new.data
    else:
        x.data[start:stop] = new

def sparse_to_numpy(x):
    return x.to_dense()

def clone(x):
    # dumb way to do this
    return x + 0

def reduce_sum(x):
    return F.sum(x)

def softmax(x, dim):
    return F.softmax(x, dim)

def spmm(x, y):
    return F.sparse_matmul(x, y)

add = operator.add
sub = operator.sub
mul = operator.mul
div = operator.truediv

def gt0(x):
    return chainer.as_variable(x.data > 0)

def sum(x, dim):
    return F.sum(x, dim)

def max(x, dim):
    return F.max(x, dim)

def min(x, dim):
    return F.min(x, dim)

def prod(x, dim):
    return F.prod(x, dim)

class record_grad(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

no_grad = chainer.no_backprop_mode
