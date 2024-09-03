import os
import tempfile
import unittest
from functools import partial

import backend as F

import numpy as np
import pytest
import torch

from dgl import graphbolt as gb


def to_on_disk_numpy(test_dir, name, t):
    path = os.path.join(test_dir, name + ".npy")
    t = t.numpy()
    np.save(path, t)
    return path


assert_equal = partial(torch.testing.assert_close, rtol=0, atol=0)


@unittest.skipIf(
    not torch.ops.graphbolt.detect_io_uring(),
    reason="DiskBasedFeature is not available on this system.",
)
def test_disk_based_feature():
    with tempfile.TemporaryDirectory() as test_dir:
        a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        b = torch.tensor([[[1, 2], [3, 4]], [[4, 5], [6, 7]]])
        c = torch.randn([4111, 47])
        metadata = {"max_value": 3}
        path_a = to_on_disk_numpy(test_dir, "a", a)
        path_b = to_on_disk_numpy(test_dir, "b", b)
        path_c = to_on_disk_numpy(test_dir, "c", c)

        feature_a = gb.DiskBasedFeature(path=path_a, metadata=metadata)
        feature_b = gb.DiskBasedFeature(path=path_b)
        feature_c = gb.DiskBasedFeature(path=path_c)

        # Read the entire feature.
        assert_equal(feature_a.read(), torch.tensor([[1, 2, 3], [4, 5, 6]]))

        assert_equal(
            feature_b.read(), torch.tensor([[[1, 2], [3, 4]], [[4, 5], [6, 7]]])
        )

        # Test read the feature with ids.
        assert_equal(
            feature_a.read(torch.tensor([0])),
            torch.tensor([[1, 2, 3]]),
        )
        assert_equal(
            feature_b.read(torch.tensor([1])),
            torch.tensor([[[4, 5], [6, 7]]]),
        )

        # Test reading into pin_memory
        if F._default_context_str == "gpu":
            res = feature_a.read(torch.tensor([0], pin_memory=True))
            assert res.is_pinned()

        # Test when the index tensor is large.
        torch_based_feature_a = gb.TorchBasedFeature(a)
        ind_a = torch.randint(low=0, high=a.size(0), size=(4111,))
        assert_equal(
            feature_a.read(ind_a),
            torch_based_feature_a.read(ind_a),
        )

        # Test converting to torch_based_feature with read_into_memory()
        torch_based_feature_b = feature_b.read_into_memory()
        ind_b = torch.randint(low=0, high=b.size(0), size=(4111,))
        assert_equal(
            feature_b.read(ind_b),
            torch_based_feature_b.read(ind_b),
        )

        # Test with larger stored feature tensor
        ind_c = torch.randint(low=0, high=c.size(0), size=(4111,))
        assert_equal(feature_c.read(ind_c), c[ind_c])

        # Test get the size and count of the entire feature.
        assert feature_a.size() == torch.Size([3])
        assert feature_b.size() == torch.Size([2, 2])
        assert feature_a.count() == a.size(0)
        assert feature_b.count() == b.size(0)

        # Test get metadata of the feature.
        assert feature_a.metadata() == metadata
        assert feature_b.metadata() == {}

        with pytest.raises(IndexError):
            feature_a.read(torch.tensor([0, 1, 2, 3]))

        # Test loading a Fortran contiguous ndarray.
        a_T = np.asfortranarray(a)
        path_a_T = test_dir + "a_T.npy"
        np.save(path_a_T, a_T)
        with pytest.raises(
            AssertionError,
            match="DiskBasedFeature only supports C_CONTIGUOUS array.",
        ):
            gb.DiskBasedFeature(path=path_a_T, metadata=metadata)

        # For windows, the file is locked by the numpy.load. We need to delete
        # it before closing the temporary directory.
        a = b = c = None
        feature_a = feature_b = feature_c = None


@unittest.skipIf(
    not torch.ops.graphbolt.detect_io_uring(),
    reason="DiskBasedFeature is not available on this system.",
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
            torch.randint(0, 127, shape, dtype=torch.float64),
            torch.randint(0, 127, shape, dtype=torch.float64),
        )
    else:
        tensor = torch.randint(0, 127, shape, dtype=dtype)
    test_tensor = tensor.clone()
    idx = torch.tensor(index, dtype=idtype)

    with tempfile.TemporaryDirectory() as test_dir:
        path = to_on_disk_numpy(test_dir, "tensor", tensor)

        feature = gb.DiskBasedFeature(path=path)

        # Test read feature.
        assert_equal(feature.read(idx), test_tensor[idx.long()])


@unittest.skipIf(
    not torch.ops.graphbolt.detect_io_uring(),
    reason="DiskBasedFeature is not available on this system.",
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
