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
    with open(npy_path, "rb") as f:
        # Read the magic string of the .npy file
        magic_str = f.read(6)
        # Verify the magic string to confirm it"s a .npy file
        if magic_str != b"\x93NUMPY":
            raise ValueError("Not a valid .npy file")

        # Read the version number of the .npy file
        version_major, _ = np.frombuffer(f.read(2), dtype=np.uint8)
        # Determine the length of the header
        if version_major == 1:
            header_len_size = 2  # version 1.x uses 2 bytes for header length
        elif version_major == 2:
            header_len_size = 4  # version 2.x uses 4 bytes for header length
        else:
            raise ValueError("Unsupported version of .npy file")

        # Read the header
        header_len = int(
            np.frombuffer(f.read(header_len_size), dtype=np.uint16)
        )
        header = f.read(header_len).decode("latin1")

        # Extract shape information from the header
        loc = header.find("(")
        loc_end = header.find(")")
        shape_str = header[loc + 1 : loc_end].replace(" ", "").split(",")

        # If there"s a trailing comma for one-dimensional arrays, remove it
        if shape_str[-1] == "":
            shape_str = shape_str[:-1]

        shape = tuple(map(int, shape_str))

        return len(shape)
