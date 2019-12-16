from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix


def cuda():
    return '/gpu:0'


def is_cuda_available():
    return tf.test.is_gpu_available(cuda_only=True)


def array_equal(a, b):
    return np.array_equal(a.numpy(), b.numpy())


def allclose(a, b, rtol=1e-4, atol=1e-4):
    return np.allclose(a.numpy(),
                       b.numpy(), rtol=rtol, atol=atol)


def randn(shape):
    return tf.random.normal(shape)


class GradContext:
    def __init__(self):
        self.tensor_for_grad = []
        self.grad_list = []
        self.tape = None

    def set_tape(self, tape):
        self.tape = tape

    def add_tensor(self, x):
        idx_pop = []
        for idx, ele in enumerate(self.tensor_for_grad):
            if ele._id == x._id:
                idx_pop.append(idx)
        if len(idx_pop) > 0:
            self.tensor_for_grad.pop(idx_pop[0])
        if self.tape is not None:
            self.tape.watch(x)
        self.tensor_for_grad.append(x)

    def backward(self, x, head_gradient=None):
        if head_gradient is not None:
            x = x * head_gradient
        self.grad_list = self.tape.gradient(x, self.tensor_for_grad)

    def is_no_grad(self, x):
        idx_pop = []
        for idx, ele in enumerate(self.tensor_for_grad):
            if ele._id == x._id:
                idx_pop.append(idx)
        if len(idx_pop) == 0:
            return True
        else:
            return self.grad_list[idx_pop[0]] is None

    def grad(self, x):
        idx_pop = []
        for idx, ele in enumerate(self.tensor_for_grad):
            if ele._id == x._id:
                idx_pop.append(idx)
        assert len(idx_pop) == 1
        t = self.grad_list[idx_pop[0]]
        return tf.convert_to_tensor(t)


cgrad = GradContext()


def get_cgrad():
    return cgrad


class record_grad:
    def __init__(self):
        self.tape = tf.GradientTape()

    def __enter__(self):
        cgrad.set_tape(self.tape)
        self.tape.__enter__()
        for x in cgrad.tensor_for_grad:
            self.tape.watch(x)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # pass
        self.tape.__exit__(exc_type, exc_value, exc_traceback)
        cgrad.tape = None


def attach_grad(x):
    cgrad.add_tensor(x)
    return x


def backward(x, head_gradient=None):
    cgrad.backward(x, head_gradient)


def grad(x):
    return cgrad.grad(x)

def is_no_grad(x):
    return cgrad.is_no_grad(x)


def full(shape, fill_value, dtype, ctx):
    with tf.device(ctx):
        t = tf.constant(fill_value, shape=shape, dtype=dtype)
    return t


def narrow_row_set(x, start, stop, new):
    # x[start:stop] = new
    raise NotImplementedError("TF doesn't support inplace update")


def sparse_to_numpy(x):
    # tf.sparse.to_dense assume sorted indices, need to turn off validate_indices in our cases
    return tf.sparse.to_dense(x, validate_indices=False).numpy()


def clone(x):
    return tf.identity(x)


def reduce_sum(x):
    return tf.reduce_sum(x)


def softmax(x, dim):
    return tf.math.softmax(x, axis=dim)


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
    return tf.reduce_sum(x, axis=dim)


def max(x, dim):
    return tf.reduce_max(x, axis=dim)


def min(x, dim):
    return tf.reduce_min(x, axis=dim)


def prod(x, dim):
    return tf.reduce_prod(x, axis=dim)


def matmul(a, b):
    return tf.linalg.matmul(a, b)


def dot(a, b):
    return sum(mul(a, b), dim=-1)


no_grad = None
