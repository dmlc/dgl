"""Utilities for DGL sparse module."""
from numbers import Number
from typing import Union

import torch


def is_scalar(x):
    """Check if the input is a scalar."""
    return isinstance(x, Number) or (torch.is_tensor(x) and x.dim() == 0)


def device_check(mat, tensor):
    """Check the input tensor device is the same as sparse matrix device.
    If not, raise an error."""
    for t in tensor:
        if mat.device != t.device:
            raise RuntimeError(
                f"indices should be on the same device as the "
                f"sparse matrix ({mat.device})"
            )


# Scalar type annotation
Scalar = Union[Number, torch.Tensor]
