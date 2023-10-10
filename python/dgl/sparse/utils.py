"""Utilities for DGL sparse module."""
from numbers import Number
from typing import Union

import torch


def is_scalar(x):
    """Check if the input is a scalar."""
    return isinstance(x, Number) or (torch.is_tensor(x) and x.dim() == 0)


# Scalar type annotation
Scalar = Union[Number, torch.Tensor]
