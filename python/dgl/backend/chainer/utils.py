import numpy as np
from chainer.backend import get_array_module as _get_array_module
try:
    import cupy
except ImportError:
    cupy = None

def is_cpu(ctx):
    return ctx.startswith('@numpy')

def is_cuda(ctx):
    return ctx.startswith('@cupy')

def get_context_module(ctx):
    if is_cpu(ctx):
        return np
    elif is_cuda(ctx):
        return cupy
    else:
        raise TypeError('Unknown device %s.' % ctx)

def get_array_module(var):
    # get_array_module(var) doesn't work
    return _get_array_module(var.data)
