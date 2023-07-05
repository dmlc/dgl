import numpy as np
import torch


def _read_torch_data(path):
    return torch.load(path)


def _read_numpy_data(path, in_memory=True):
    if in_memory:
        return torch.from_numpy(np.load(path))
    return torch.as_tensor(np.load(path, mmap_mode="r+"))


def read_data(path, format, in_memory=True):
    if format == "torch":
        return _read_torch_data(path)
    elif format == "numpy":
        return _read_numpy_data(path, in_memory=in_memory)
    else:
        raise RuntimeError("Unsupported format: {}".format(format))


def tensor_to_tuple(data):
    assert isinstance(data, torch.Tensor), "data must be a torch.Tensor"
    return tuple(data.t())
