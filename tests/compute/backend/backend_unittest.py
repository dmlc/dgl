"""This file defines the unified tensor framework interface required by DGL
unit testing, other than the ones used in the framework itself.
"""

###############################################################################
# Tensor, data type and context interfaces

def cuda():
    """Context object for CUDA."""
    pass

###############################################################################
# Tensor functions on feature data
# --------------------------------
# These functions are performance critical, so it's better to have efficient
# implementation in each framework.

def equal(a, b):
    """Check whether the two tensors are *exactly* equal."""
    pass

def allclose(a, b):
    """Check whether the two tensors are numerically close to each other."""
    pass

def randn(shape):
    """Generate a tensor with elements from standard normal distribution."""
    pass

def attach_grad(x):
    """Flag the tensor *in-place* to have its gradient computed in backward
    pass."""
    pass

def backward(x, head_gradient=None):
    """Invoke backward computation with an optional head gradient.
    
    Returns nothing."""
    pass

def grad(x):
    """Fetches the gradient from the tensor after backward computation."""
    pass

def full(shape, fill_value, dtype, ctx):
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

class record_grad(object):
    """Context manager that records the gradients"""
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass


class no_grad(object):
    """Context manager that explicitly disables gradient computation"""
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
