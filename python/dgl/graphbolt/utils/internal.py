"""Utility functions for GraphBolt."""

import os

import numpy as np
from numpy.lib.format import read_array_header_1_0, read_array_header_2_0
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
    if fmt == "numpy" and isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    elif fmt == "torch" and isinstance(data, np.ndarray):
        data = torch.from_numpy(data).cpu()

    # Save the data.
    if fmt == "numpy":
        np.save(path, data)
    elif fmt == "torch":
        torch.save(data, path)


def get_npy_dim(npy_path):
    """Get the dim of numpy file."""
    with open(npy_path, "rb") as f:
        magic_str = f.read(6)
        if magic_str != b'\x93NUMPY':
            raise ValueError("Not a .npy file")
        # Use the corresponding version of func to get header. 
        version = f.read(2)
        if version == b"\x01\x00":
            header, _, _ = read_array_header_1_0(f)
        elif version == b"\x02\x00":
            header, _, _ = read_array_header_2_0(f)
        else:
            raise ValueError(f"Unsupported .npy version")
        
        return len(header)
