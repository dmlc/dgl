"""This file defines the unified tensor framework interface required by DGL
unit testing, other than the ones used in the framework itself.
"""

###############################################################################
# Tensor, data type and context interfaces


def cuda():
    """Context object for CUDA."""
    pass


def is_cuda_available():
    """Check whether CUDA is available."""
    pass


###############################################################################
# Tensor functions on feature data
# --------------------------------
# These functions are performance critical, so it's better to have efficient
# implementation in each framework.


def array_equal(a, b):
    """Check whether the two tensors are *exactly* equal."""
    pass


def allclose(a, b, rtol=1e-4, atol=1e-4):
    """Check whether the two tensors are numerically close to each other."""
    pass


def randn(shape):
    """Generate a tensor with elements from standard normal distribution."""
    pass


def full(shape, fill_value, dtype, ctx):
    pass


def narrow_row_set(x, start, stop, new):
    """Set a slice of the given tensor to a new value."""
    pass


def sparse_to_numpy(x):
    """Convert a sparse tensor to a numpy array."""
    pass


def clone(x):
    pass


def reduce_sum(x):
    """Sums all the elements into a single scalar."""
    pass


def softmax(x, dim):
    """Softmax Operation on Tensors"""
    pass


def spmm(x, y):
    """Sparse dense matrix multiply"""
    pass


def add(a, b):
    """Compute a + b"""
    pass


def sub(a, b):
    """Compute a - b"""
    pass


def mul(a, b):
    """Compute a * b"""
    pass


def div(a, b):
    """Compute a / b"""
    pass


def sum(x, dim, keepdims=False):
    """Computes the sum of array elements over given axes"""
    pass


def max(x, dim):
    """Computes the max of array elements over given axes"""
    pass


def min(x, dim):
    """Computes the min of array elements over given axes"""
    pass


def prod(x, dim):
    """Computes the prod of array elements over given axes"""
    pass


def matmul(a, b):
    """Compute Matrix Multiplication between a and b"""
    pass


def dot(a, b):
    """Compute Dot between a and b"""
    pass


def abs(a):
    """Compute the absolute value of a"""
    pass


def seed(a):
    """Set seed to for random generator"""
    pass


###############################################################################
# Tensor functions used *only* on index tensor
# ----------------
# These operators are light-weighted, so it is acceptable to fallback to
# numpy operators if currently missing in the framework. Ideally in the future,
# DGL should contain all the operations on index, so this set of operators
# should be gradually removed.

###############################################################################
# Other interfaces
# ----------------
# These are not related to tensors. Some of them are temporary workarounds that
# should be included in DGL in the future.
