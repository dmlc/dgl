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
        b = torch.tensor([[1, 2, 3], [4, 5, 6]])
        if not in_memory:
            a = to_on_disk_tensor(test_dir, "a", a)
            b = to_on_disk_tensor(test_dir, "b", b)

        feat_store_a = gb.TorchBasedFeatureStore(a)
        feat_store_b = gb.TorchBasedFeatureStore(b)

        assert torch.equal(feat_store_a.read(), torch.tensor([1, 2, 3]))
        assert torch.equal(
            feat_store_b.read(), torch.tensor([[1, 2, 3], [4, 5, 6]])
        )
        assert torch.equal(
            feat_store_a.read(torch.tensor([0, 2])),
            torch.tensor([1, 3]),
        )
        assert torch.equal(
            feat_store_a.read(torch.tensor([1, 1])),
            torch.tensor([2, 2]),
        )
        assert torch.equal(
            feat_store_b.read(torch.tensor([1])),
            torch.tensor([[4, 5, 6]]),
        )
        feat_store_a.update(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
        assert torch.equal(feat_store_a.read(), torch.tensor([0, 1, 2]))
        feat_store_a.update(torch.tensor([2, 0]), torch.tensor([0, 2]))
        assert torch.equal(feat_store_a.read(), torch.tensor([2, 1, 0]))

        with pytest.raises(IndexError):
            feat_store_a.read(torch.tensor([0, 1, 2, 3]))

        # For windows, the file is locked by the numpy.load. We need to delete
        # it before closing the temporary directory.
        a = b = None
        feat_store_a = feat_store_b = None


def write_tensor_to_disk(dir, name, t, fmt="pt"):
    if fmt == "pt":
        torch.save(t, os.path.join(dir, name + ".pt"))
    else:
        t = t.numpy()
        np.save(os.path.join(dir, name + ".npy"), t)


@pytest.mark.parametrize("in_memory", [True, False])
def test_load_feature_stores(in_memory):
    with tempfile.TemporaryDirectory() as test_dir:
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([2, 5, 3])
        write_tensor_to_disk(test_dir, "a", a, fmt="pt")
        write_tensor_to_disk(test_dir, "b", b, fmt="npy")
        feat_data = [
            gb.OnDiskFeatureData(
                name="a",
                format="torch",
                path=os.path.join(test_dir, "a.pt"),
                in_memory=True,
            ),
            gb.OnDiskFeatureData(
                name="b",
                format="numpy",
                path=os.path.join(test_dir, "b.npy"),
                in_memory=in_memory,
            ),
        ]
        feat_stores = gb.load_feature_stores(feat_data)
        assert torch.equal(feat_stores["a"].read(), torch.tensor([1, 2, 3]))
        assert torch.equal(feat_stores["b"].read(), torch.tensor([2, 5, 3]))

        # For windows, the file is locked by the numpy.load. We need to delete
        # it before closing the temporary directory.
        a = b = None
        feat_stores = None
