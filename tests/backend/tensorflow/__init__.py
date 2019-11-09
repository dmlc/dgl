from __future__ import absolute_import

import numpy as np
import tensorflow as tf

def cuda():
    return tf.device('/gpu:0')

def is_cuda_available():
    return tf.test.is_gpu_available(cuda_only=True)

def array_equal(a, b):
    return np.array_equal(a.numpy(), b.numpy())

def allclose(a, b, rtol=1e-4, atol=1e-4):
    return np.allclose(a.numpy(),
            b.numpy(), rtol=rtol, atol=atol)

def randn(shape):
    return tf.random.normal(shape)

class record_grad(object):
    def __init__(self):
        self.tensor_for_grad=[]
        self.grads= []
    
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
    
    def add_tensor(self, x):
        self.tensor_for_grad.append(x)
    
    def backward(self, x):
        with tf.GradientTape() as tape:
            self.grads = tape.gradient(x, self.tensor_for_grad)
    
    def is_no_grad(self, x):
        return x in self.tensor_for_grad
    
    def grad(self, x):
        return self.grads[self.tensor_for_grad.index(x)]


def attach_grad(x):
    if x.grad is not None:
        x.grad.zero_()
        return x
    else:
        return x.requires_grad_()

def backward(x, head_gradient=None):
    x.backward(head_gradient)

def grad(x):
    return x.grad

def is_no_grad(x):
    return x.grad is None or (x.grad == 0).all()

def full(shape, fill_value, dtype, ctx):
    return th.full(shape, fill_value, dtype=dtype, device=ctx)

def narrow_row_set(x, start, stop, new):
    # x[start:stop] = new
    raise NotImplementedError("TF doesn't support inplace update")

def sparse_to_numpy(x):
    return tf.sparse.to_dense(input).numpy()

def clone(x):
    return tf.identity(x)

def reduce_sum(x):
    return tf.reduce_sum(x)

def softmax(x, dim):
    return tf.math.softmax(input, axis=dim)

def spmm(x, y):
    return tf.sparse.sparse_dense_matmul(x, y)

def add(a, b):
    return a + b

def sub(a, b):
    return a - b

def mul(a, b):
    return a * b

def div(a, b):
    return a / b

def sum(x, dim):
    return tf.reduce_sum(input, axis=dim)

def max(x, dim):
    return tf.reduce_max(input, axis=dim)

def min(x, dim):
    return tf.reduce_min(input, axis=dim)

def prod(x, dim):
    return tf.reduce_prod(input, axis=dim)

def matmul(a, b):
    return tf.linalg.matmul(a, b)

def dot(a, b):
    return sum(mul(a, b), dim=-1)


no_grad = None
