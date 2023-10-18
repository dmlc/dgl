"""Utility functions for GraphBolt."""

import os

import numpy as np
import torch
from numpy.lib.format import read_array_header_1_0, read_array_header_2_0


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
    if fmt == "numpy" and isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    elif fmt == "torch" and isinstance(data, np.ndarray):
        data = torch.from_numpy(data).cpu()

    # Save the data.
    if fmt == "numpy":
        if not data.flags["C_CONTIGUOUS"]:
            Warning(
                "The ndarray saved to disk is not contiguous, "
                "so it will be copied to contiguous memory."
            )
            data = np.ascontiguousarray(data)
        np.save(path, data)
    elif fmt == "torch":
        if not data.is_contiguous():
            Warning(
                "The tensor saved to disk is not contiguous, "
                "so it will be copied to contiguous memory."
            )
            data = data.contiguous()
        torch.save(data, path)


def get_npy_dim(npy_path):
    """Get the dim of numpy file."""
    with open(npy_path, "rb") as f:
        # For the read_array_header API provided by numpy will only read the
        # length of the header, it will cause parsing failure and error if
        # first 8 bytes which contains magin string and version are not read
        # ahead of time. So, we need to make sure we have skipped these 8
        # bytes.
        f.seek(8, 0)
        try:
            shape, _, _ = read_array_header_1_0(f)
        except ValueError:
            try:
                shape, _, _ = read_array_header_2_0(f)
            except ValueError:
                raise ValueError("Invalid file format")

        return len(shape)
