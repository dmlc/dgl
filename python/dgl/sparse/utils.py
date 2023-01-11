"""DGL sparse utility module."""
from numbers import Number
import torch

def is_scalar(x):
    """Check if the input is a scalar.
    """
    return isinstance(x, Number) or (torch.is_tensor(x) and x.dim() == 0)
