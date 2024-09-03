"""Utility functions for GraphBolt."""

import hashlib
import json
import os
import shutil
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import read_array_header_1_0, read_array_header_2_0


def numpy_save_aligned(*args, **kwargs):
    """A wrapper for numpy.save(), ensures the array is stored 4KiB aligned."""
    # https://github.com/numpy/numpy/blob/2093a6d5b933f812d15a3de0eafeeb23c61f948a/numpy/lib/format.py#L179
    has_array_align = hasattr(np.lib.format, "ARRAY_ALIGN")
    if has_array_align:
        default_alignment = np.lib.format.ARRAY_ALIGN
        # The maximum allowed alignment by the numpy code linked above is 4K.
        # Most filesystems work with block sizes of 4K so in practice, the file
        # size on the disk won't be larger.
        np.lib.format.ARRAY_ALIGN = 4096
    np.save(*args, **kwargs)
    if has_array_align:
        np.lib.format.ARRAY_ALIGN = default_alignment


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
        numpy_save_aligned(path, data)
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


def _to_int32(data):
    if isinstance(data, torch.Tensor):
        return data.to(torch.int32)
    elif isinstance(data, np.ndarray):
        return data.astype(np.int32)
    else:
        raise TypeError(
            "Unsupported input type. Please provide a torch tensor or numpy array."
        )


def copy_or_convert_data(
    input_path,
    output_path,
    input_format,
    output_format="numpy",
    in_memory=True,
    is_feature=False,
    within_int32=False,
):
    """Copy or convert the data from input_path to output_path."""
    assert (
        output_format == "numpy"
    ), "The output format of the data should be numpy."
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # We read the data always in case we need to cast its type.
    data = read_data(input_path, input_format, in_memory)
    if within_int32:
        data = _to_int32(data)
    if input_format == "numpy":
        # If dim of the data is 1, reshape it to n * 1 and save it to output_path.
        if is_feature and get_npy_dim(input_path) == 1:
            data = data.reshape(-1, 1)
        # If the data does not need to be modified, just copy the file.
        elif not within_int32 and data.numpy().flags["C_CONTIGUOUS"]:
            shutil.copyfile(input_path, output_path)
            return
    else:
        # If dim of the data is 1, reshape it to n * 1 and save it to output_path.
        if is_feature and data.dim() == 1:
            data = data.reshape(-1, 1)
    save_data(data, output_path, output_format)


def read_edges(dataset_dir, edge_fmt, edge_path):
    """Read egde data from numpy or csv."""
    assert edge_fmt in [
        "numpy",
        "csv",
    ], f"`numpy` or `csv` is expected when reading edges but got `{edge_fmt}`."
    if edge_fmt == "numpy":
        edge_data = read_data(
            os.path.join(dataset_dir, edge_path),
            edge_fmt,
        )
        assert (
            edge_data.shape[0] == 2 and len(edge_data.shape) == 2
        ), f"The shape of edges should be (2, N), but got {edge_data.shape}."
        src, dst = edge_data.numpy()
    else:
        edge_data = pd.read_csv(
            os.path.join(dataset_dir, edge_path),
            names=["src", "dst"],
        )
        src, dst = edge_data["src"].to_numpy(), edge_data["dst"].to_numpy()
    return (src, dst)


def calculate_file_hash(file_path, hash_algo="md5"):
    """Calculate the hash value of a file."""
    hash_algos = ["md5", "sha1", "sha224", "sha256", "sha384", "sha512"]
    if hash_algo in hash_algos:
        hash_obj = getattr(hashlib, hash_algo)()
    else:
        raise ValueError(
            f"Hash algorithm must be one of: {hash_algos}, but got `{hash_algo}`."
        )
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def calculate_dir_hash(
    dir_path, hash_algo="md5", ignore: Union[str, List[str]] = None
):
    """Calculte the hash values of all files under the directory."""
    hashes = {}
    for dirpath, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if ignore and filename in ignore:
                continue
            filepath = os.path.join(dirpath, filename)
            file_hash = calculate_file_hash(filepath, hash_algo=hash_algo)
            hashes[filepath] = file_hash
    return hashes


def check_dataset_change(dataset_dir, processed_dir):
    """Check whether dataset has been changed by checking its hash value."""
    hash_value_file = "dataset_hash_value.txt"
    hash_value_file_path = os.path.join(
        dataset_dir, processed_dir, hash_value_file
    )
    if not os.path.exists(hash_value_file_path):
        return True
    with open(hash_value_file_path, "r") as f:
        oringinal_hash_value = json.load(f)
    present_hash_value = calculate_dir_hash(dataset_dir, ignore=hash_value_file)
    if oringinal_hash_value == present_hash_value:
        force_preprocess = False
    else:
        force_preprocess = True
    return force_preprocess
