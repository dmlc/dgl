import os
import tempfile

import numpy as np
import pytest
import torch
from dgl import graphbolt as gb


def to_on_disk_tensor(test_dir, name, t):
    path = os.path.join(test_dir, name + ".npy")
    t = t.numpy()
    np.save(path, t)
    # The Pytorch tensor is a view of the numpy array on disk, which does not
    # consume memory.
    t = torch.as_tensor(np.load(path, mmap_mode="r+"))
    return t


@pytest.mark.parametrize("in_memory", [True, False])
def test_torch_based_feature_store(in_memory):
    with tempfile.TemporaryDirectory() as test_dir:
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([3, 4, 5])
        c = torch.tensor([[1, 2, 3], [4, 5, 6]])
        if not in_memory:
            a = to_on_disk_tensor(test_dir, "a", a)
            b = to_on_disk_tensor(test_dir, "b", b)
            c = to_on_disk_tensor(test_dir, "c", c)

        feature_store = gb.TorchBasedFeatureStore({"a": a, "b": b, "c": c})
        assert torch.equal(feature_store.read("a"), torch.tensor([1, 2, 3]))
        assert torch.equal(feature_store.read("b"), torch.tensor([3, 4, 5]))
        assert torch.equal(
            feature_store.read("a", torch.tensor([0, 2])),
            torch.tensor([1, 3]),
        )
        assert torch.equal(
            feature_store.read("a", torch.tensor([1, 1])),
            torch.tensor([2, 2]),
        )
        assert torch.equal(
            feature_store.read("c", torch.tensor([1])),
            torch.tensor([[4, 5, 6]]),
        )
        feature_store.update("a", torch.tensor([0, 1, 2]))
        assert torch.equal(feature_store.read("a"), torch.tensor([0, 1, 2]))
        assert torch.equal(
            feature_store.read("a", torch.tensor([0, 2])),
            torch.tensor([0, 2]),
        )
        with pytest.raises(AssertionError):
            feature_store.read("d")

        with pytest.raises(IndexError):
            feature_store.read("a", torch.tensor([0, 1, 2, 3]))

        # For windows, the file is locked by the numpy.load. We need to delete
        # it before closing the temporary directory.
        a = b = c = feature_store = None
