import os
import sys
import tempfile
import unittest

import numpy as np
import pytest
import torch

from dgl import graphbolt as gb


def to_on_disk_numpy(test_dir, name, t):
    path = os.path.join(test_dir, name + ".npy")
    t = t.numpy()
    np.save(path, t)
    return path


@unittest.skipIf(
    sys.platform.startswith("win"),
    reason="Tests for disk dataset can only deployed on Linux,"
    "because the io_uring is only supportted by Linux kernel.",
)
def test_disk_based_feature():
    with tempfile.TemporaryDirectory() as test_dir:
        a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        b = torch.tensor([[[1, 2], [3, 4]], [[4, 5], [6, 7]]])
        metadata = {"max_value": 3}
        path_a = to_on_disk_numpy(test_dir, "a", a)
        path_b = to_on_disk_numpy(test_dir, "b", b)

        feature_a = gb.DiskBasedFeature(path=path_a, metadata=metadata)
        feature_b = gb.DiskBasedFeature(path=path_b)

        # Read the entire feature.
        assert torch.equal(
            feature_a.read(), torch.tensor([[1, 2, 3], [4, 5, 6]])
        )

        # Test read the feature with ids.
        assert torch.equal(
            feature_b.read(), torch.tensor([[[1, 2], [3, 4]], [[4, 5], [6, 7]]])
        )

        # Read the feature with ids.
        assert torch.equal(
            feature_a.read(torch.tensor([0])),
            torch.tensor([[1, 2, 3]]),
        )
        assert torch.equal(
            feature_b.read(torch.tensor([1])),
            torch.tensor([[[4, 5], [6, 7]]]),
        )

        # Test get the size of the entire feature.
        assert feature_a.size() == torch.Size([3])
        assert feature_b.size() == torch.Size([2, 2])

        # Test get metadata of the feature.
        assert feature_a.metadata() == metadata
        assert feature_b.metadata() == {}

        with pytest.raises(IndexError):
            feature_a.read(torch.tensor([0, 1, 2, 3]))

        # For windows, the file is locked by the numpy.load. We need to delete
        # it before closing the temporary directory.
        a = b = None
        feature_a = feature_b = None

        # Test loaded tensors' contiguity from C/Fortran contiguous ndarray.
        contiguous_numpy = np.array([[1, 2, 3], [4, 5, 6]], order="C")
        non_contiguous_numpy = np.array([[1, 2, 3], [4, 5, 6]], order="F")
        assert contiguous_numpy.flags["C_CONTIGUOUS"]
        assert non_contiguous_numpy.flags["F_CONTIGUOUS"]

        path_contiguous = os.path.join(test_dir, "contiguous_numpy.npy")
        path_non_contiguous = os.path.join(test_dir, "non_contiguous_numpy.npy")
        np.save(path_contiguous, contiguous_numpy)
        np.save(path_non_contiguous, non_contiguous_numpy)

        feature_c = gb.DiskBasedFeature(path=path_contiguous, metadata=metadata)
        feature_n = gb.DiskBasedFeature(path=path_non_contiguous)

        assert feature_c._tensor.is_contiguous()
        assert feature_n._tensor.is_contiguous()

        contiguous_numpy = non_contiguous_numpy = None
        feature_c = feature_n = None


@unittest.skipIf(
    sys.platform.startswith("win"),
    reason="Tests for disk dataset can only deployed on Linux,"
    "because the io_uring is only supportted by Linux kernel.",
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
        torch.int8,
        torch.float16,
        torch.complex128,
    ],
)
@pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "shape", [(10, 20), (20, 10), (20, 25, 10), (137, 50, 30)]
)
@pytest.mark.parametrize("index", [[0], [1, 2, 3], [0, 6, 2, 8]])
def test_more_disk_based_feature(dtype, idtype, shape, index):
    if dtype == torch.complex128:
        tensor = torch.complex(
            torch.randint(0, 13, shape, dtype=torch.float64),
            torch.randint(0, 13, shape, dtype=torch.float64),
        )
    else:
        tensor = torch.randint(0, 13, shape, dtype=dtype)
    test_tensor = tensor.clone()
    idx = torch.tensor(index)

    with tempfile.TemporaryDirectory() as test_dir:
        path = to_on_disk_numpy(test_dir, "tensor", tensor)

        feature = gb.DiskBasedFeature(path=path)

        # Test read feature.
        assert torch.equal(
            feature.read(torch.tensor(idx, dtype=idtype)), test_tensor[idx]
        )


@unittest.skipIf(
    sys.platform.startswith("win"),
    reason="Tests for large disk dataset can only deployed on Linux,"
    "because the io_uring is only supportted by Linux kernel.",
)
def test_disk_based_feature_repr():
    with tempfile.TemporaryDirectory() as test_dir:
        a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        b = torch.tensor([[[1, 2], [3, 4]], [[4, 5], [6, 7]]])
        metadata = {"max_value": 3}

        path_a = to_on_disk_numpy(test_dir, "a", a)
        path_b = to_on_disk_numpy(test_dir, "b", b)

        feature_a = gb.DiskBasedFeature(path=path_a, metadata=metadata)
        feature_b = gb.DiskBasedFeature(path=path_b)

        expected_str_feature_a = str(
            "DiskBasedFeature(\n"
            "    feature=tensor([[1, 2, 3],\n"
            "                    [4, 5, 6]]),\n"
            "    metadata={'max_value': 3},\n"
            ")"
        )
        expected_str_feature_b = str(
            "DiskBasedFeature(\n"
            "    feature=tensor([[[1, 2],\n"
            "                     [3, 4]],\n"
            "\n"
            "                    [[4, 5],\n"
            "                     [6, 7]]]),\n"
            "    metadata={},\n"
            ")"
        )
        assert str(feature_a) == expected_str_feature_a
        assert str(feature_b) == expected_str_feature_b
        a = b = metadata = None
        feature_a = feature_b = None
        expected_str_feature_a = expected_str_feature_b = None
