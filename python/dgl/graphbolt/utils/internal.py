"""Utility functions for GraphBolt."""

import os

import numpy as np
import torch


def _read_torch_data(path):
    return torch.load(path)


def _read_numpy_data(path, in_memory=True):
    if in_memory:
        return torch.from_numpy(np.load(path))
    return torch.as_tensor(np.load(path, mmap_mode="r+"))


def read_data(path, fmt, in_memory=True):
    """Read data from disk."""
    if fmt == "torch":
        return _read_torch_data(path)
    elif fmt == "numpy":
        return _read_numpy_data(path, in_memory=in_memory)
    else:
        raise RuntimeError(f"Unsupported format: {fmt}")


def save_data(data, path, fmt):
    """Save data into disk."""
    # Make sure the directory exists.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if fmt not in ["numpy", "torch"]:
        raise RuntimeError(f"Unsupported format: {fmt}")

    # Perform necessary conversion.
    if fmt == "numpy" and isinstance(element, torch.Tensor):
        element = element.cpu()
        if not element.is_contiguous():
            Warning(
                "The ndarray saved to disk is not contiguous, "
                "so it will be copied to contiguous memory."
            )
            element = element.contiguous()
        element = element.numpy()
    elif fmt == "torch" and isinstance(element, np.ndarray):
        element = torch.from_numpy(element).cpu()
        if not element.is_contiguous():
            Warning(
                "The tensor saved to disk is not contiguous, "
                "so it will be copied to contiguous memory."
            )
            element = element.contiguous()

    # Save the data.
    if fmt == "numpy":
        np.save(path, data)
    elif fmt == "torch":
        torch.save(data, path)
