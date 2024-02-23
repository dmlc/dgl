""" Utility functions for distributed training."""

import torch

from ..utils import toindex


def totensor(data):
    """Convert the given data to a tensor.

    Parameters
    ----------
    data : tensor, array, list or slice
        Data to be converted.

    Returns
    -------
    Tensor
        Converted tensor.
    """
    if isinstance(data, torch.Tensor):
        return data
    return toindex(data).tousertensor()
