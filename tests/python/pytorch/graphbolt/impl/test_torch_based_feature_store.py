import os
import tempfile
import unittest

import backend as F

import numpy as np
import pydantic
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
def test_torch_based_feature(in_memory):
    with tempfile.TemporaryDirectory() as test_dir:
        a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        b = torch.tensor([[[1, 2], [3, 4]], [[4, 5], [6, 7]]])
        metadata = {"max_value": 3}
        if not in_memory:
            a = to_on_disk_tensor(test_dir, "a", a)
            b = to_on_disk_tensor(test_dir, "b", b)

        feature_a = gb.TorchBasedFeature(a, metadata=metadata)
        feature_b = gb.TorchBasedFeature(b)

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
        # Update the feature with ids.
        feature_a.update(torch.tensor([[0, 1, 2]]), torch.tensor([0]))
        assert torch.equal(
            feature_a.read(), torch.tensor([[0, 1, 2], [4, 5, 6]])
        )
        feature_b.update(torch.tensor([[[1, 2], [3, 4]]]), torch.tensor([1]))
        assert torch.equal(
            feature_b.read(), torch.tensor([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
        )

        # Test update the feature.
        feature_a.update(torch.tensor([[5, 1, 3]]))
        assert torch.equal(
            feature_a.read(),
            torch.tensor([[5, 1, 3]]),
        ), print(feature_a.read())
        feature_b.update(
            torch.tensor([[[1, 3], [5, 7]], [[2, 4], [6, 8]], [[2, 4], [6, 8]]])
        )
        assert torch.equal(
            feature_b.read(),
            torch.tensor(
                [[[1, 3], [5, 7]], [[2, 4], [6, 8]], [[2, 4], [6, 8]]]
            ),
        )

        # Test get the size and count of the entire feature.
        assert feature_a.size() == torch.Size([3])
        assert feature_b.size() == torch.Size([2, 2])
        assert feature_a.count() == 1
        assert feature_b.count() == 3

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
        np.save(
            os.path.join(test_dir, "contiguous_numpy.npy"), contiguous_numpy
        )
        np.save(
            os.path.join(test_dir, "non_contiguous_numpy.npy"),
            non_contiguous_numpy,
        )

        cur_mmap_mode = None
        if not in_memory:
            cur_mmap_mode = "r+"
        feature_a = gb.TorchBasedFeature(
            torch.from_numpy(
                np.load(
                    os.path.join(test_dir, "contiguous_numpy.npy"),
                    mmap_mode=cur_mmap_mode,
                )
            )
        )
        feature_b = gb.TorchBasedFeature(
            torch.from_numpy(
                np.load(
                    os.path.join(test_dir, "non_contiguous_numpy.npy"),
                    mmap_mode=cur_mmap_mode,
                )
            )
        )
        assert feature_a._tensor.is_contiguous()
        assert feature_b._tensor.is_contiguous()

        contiguous_numpy = non_contiguous_numpy = None
        feature_a = feature_b = None


def is_feature_store_on_cuda(store):
    for feature in store._features.values():
        assert feature._tensor.is_cuda


def is_feature_store_on_cpu(store):
    for feature in store._features.values():
        assert not feature._tensor.is_cuda


@unittest.skipIf(
    F._default_context_str == "cpu",
    reason="Tests for pinned memory are only meaningful on GPU.",
)
@pytest.mark.parametrize("device", ["pinned", "cuda"])
def test_feature_store_to_device(device):
    with tempfile.TemporaryDirectory() as test_dir:
        a = torch.tensor([[1, 2, 4], [2, 5, 3]])
        b = torch.tensor([[[1, 2], [3, 4]], [[2, 5], [3, 4]]])
        write_tensor_to_disk(test_dir, "a", a, fmt="torch")
        write_tensor_to_disk(test_dir, "b", b, fmt="numpy")
        feature_data = [
            gb.OnDiskFeatureData(
                domain="node",
                type="paper",
                name="a",
                format="torch",
                path=os.path.join(test_dir, "a.pt"),
            ),
            gb.OnDiskFeatureData(
                domain="edge",
                type="paper:cites:paper",
                name="b",
                format="numpy",
                path=os.path.join(test_dir, "b.npy"),
            ),
        ]
        feature_store = gb.TorchBasedFeatureStore(feature_data)
        feature_store2 = feature_store.to(device)
        if device == "pinned":
            assert feature_store2.is_pinned()
        elif device == "cuda":
            is_feature_store_on_cuda(feature_store2)

        # The original variable should be untouched.
        is_feature_store_on_cpu(feature_store)


@unittest.skipIf(
    F._default_context_str == "cpu",
    reason="Tests for pinned memory are only meaningful on GPU.",
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
@pytest.mark.parametrize("shape", [(2, 1), (2, 3), (2, 2, 2), (137, 13, 3)])
@pytest.mark.parametrize("in_place", [False, True])
def test_torch_based_pinned_feature(dtype, idtype, shape, in_place):
    if dtype == torch.complex128:
        tensor = torch.complex(
            torch.randint(0, 13, shape, dtype=torch.float64),
            torch.randint(0, 13, shape, dtype=torch.float64),
        )
    else:
        tensor = torch.randint(0, 13, shape, dtype=dtype)
    test_tensor = tensor.clone().detach()
    test_tensor_cuda = test_tensor.cuda()

    feature = gb.TorchBasedFeature(tensor)
    if in_place:
        if gb.is_wsl():
            pytest.skip("In place pinning is not supported on WSL.")
        feature.pin_memory_()

        # Check if pinning is truly in-place.
        assert feature._tensor.data_ptr() == tensor.data_ptr()
    else:
        feature = feature.to("pinned")

    assert feature.is_pinned()

    # Test read entire pinned feature, the result should be on cuda.
    assert torch.equal(feature.read(), test_tensor_cuda)
    assert feature.read().is_cuda
    assert torch.equal(
        feature.read(torch.tensor([0], dtype=idtype).cuda()),
        test_tensor_cuda[[0]],
    )

    # Test read pinned feature with idx on cuda, the result should be on cuda.
    assert feature.read(torch.tensor([0], dtype=idtype).cuda()).is_cuda

    # Test read pinned feature with idx on cpu, the result should be on cpu.
    assert torch.equal(
        feature.read(torch.tensor([0], dtype=idtype)), test_tensor[[0]]
    )
    assert not feature.read(torch.tensor([0], dtype=idtype)).is_cuda


def write_tensor_to_disk(dir, name, t, fmt="torch"):
    if fmt == "torch":
        torch.save(t, os.path.join(dir, name + ".pt"))
    elif fmt == "numpy":
        t = t.numpy()
        np.save(os.path.join(dir, name + ".npy"), t)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


@pytest.mark.parametrize("in_memory", [True, False])
def test_torch_based_feature_store(in_memory):
    with tempfile.TemporaryDirectory() as test_dir:
        a = torch.tensor([[1, 2, 4], [2, 5, 3]])
        b = torch.tensor([[[1, 2], [3, 4]], [[2, 5], [3, 4]]])
        write_tensor_to_disk(test_dir, "a", a, fmt="torch")
        write_tensor_to_disk(test_dir, "b", b, fmt="numpy")
        feature_data = [
            gb.OnDiskFeatureData(
                domain="node",
                type="paper",
                name="a",
                format="torch",
                path=os.path.join(test_dir, "a.pt"),
                in_memory=True,
            ),
            gb.OnDiskFeatureData(
                domain="edge",
                type="paper:cites:paper",
                name="b",
                format="numpy",
                path=os.path.join(test_dir, "b.npy"),
                in_memory=in_memory,
            ),
        ]
        feature_store = gb.TorchBasedFeatureStore(feature_data)

        assert isinstance(
            feature_store[("node", "paper", "a")], gb.TorchBasedFeature
        )
        assert isinstance(
            feature_store[("edge", "paper:cites:paper", "b")],
            gb.TorchBasedFeature if in_memory else gb.DiskBasedFeature,
        )

        # Test read the entire feature.
        assert torch.equal(
            feature_store.read("node", "paper", "a"),
            torch.tensor([[1, 2, 4], [2, 5, 3]]),
        )
        assert torch.equal(
            feature_store.read("edge", "paper:cites:paper", "b"),
            torch.tensor([[[1, 2], [3, 4]], [[2, 5], [3, 4]]]),
        )

        # Test get the size of the entire feature.
        assert feature_store.size("node", "paper", "a") == torch.Size([3])
        assert feature_store.size(
            "edge", "paper:cites:paper", "b"
        ) == torch.Size([2, 2])

        # Test get the keys of the features.
        assert feature_store.keys() == [
            ("node", "paper", "a"),
            ("edge", "paper:cites:paper", "b"),
        ]

        # For windows, the file is locked by the numpy.load. We need to delete
        # it before closing the temporary directory.
        a = b = None
        feature_store = None

        # ``domain`` should be enum.
        with pytest.raises(pydantic.ValidationError):
            _ = gb.OnDiskFeatureData(
                domain="invalid",
                type="paper",
                name="a",
                format="torch",
                path=os.path.join(test_dir, "a.pt"),
                in_memory=True,
            )

        # ``type`` could be null.
        feature_data = [
            gb.OnDiskFeatureData(
                domain="node",
                name="a",
                format="torch",
                path=os.path.join(test_dir, "a.pt"),
                in_memory=True,
            ),
        ]
        feature_store = gb.TorchBasedFeatureStore(feature_data)
        # Test read the entire feature.
        assert torch.equal(
            feature_store.read("node", None, "a"),
            torch.tensor([[1, 2, 4], [2, 5, 3]]),
        )
        # Test get the size of the entire feature.
        assert feature_store.size("node", None, "a") == torch.Size([3])

        feature_store = None


@pytest.mark.parametrize("in_memory", [True, False])
def test_torch_based_feature_repr(in_memory):
    with tempfile.TemporaryDirectory() as test_dir:
        a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        b = torch.tensor([[[1, 2], [3, 4]], [[4, 5], [6, 7]]])
        metadata = {"max_value": 3}
        if not in_memory:
            a = to_on_disk_tensor(test_dir, "a", a)
            b = to_on_disk_tensor(test_dir, "b", b)

        feature_a = gb.TorchBasedFeature(a, metadata=metadata)
        feature_b = gb.TorchBasedFeature(b)

        expected_str_feature_a = (
            "TorchBasedFeature(\n"
            "    feature=tensor([[1, 2, 3],\n"
            "                    [4, 5, 6]]),\n"
            "    metadata={'max_value': 3},\n"
            ")"
        )
        expected_str_feature_b = (
            "TorchBasedFeature(\n"
            "    feature=tensor([[[1, 2],\n"
            "                     [3, 4]],\n"
            "\n"
            "                    [[4, 5],\n"
            "                     [6, 7]]]),\n"
            "    metadata={},\n"
            ")"
        )

        assert repr(feature_a) == expected_str_feature_a, feature_a
        assert repr(feature_b) == expected_str_feature_b, feature_b

        a = b = metadata = None
        feature_a = feature_b = None
        expected_str_feature_a = expected_str_feature_b = None


@pytest.mark.parametrize("in_memory", [True, False])
def test_torch_based_feature_store_repr(in_memory):
    with tempfile.TemporaryDirectory() as test_dir:
        a = torch.tensor([[1, 2, 4], [2, 5, 3]])
        b = torch.tensor([[[1, 2], [3, 4]], [[2, 5], [3, 4]]])
        write_tensor_to_disk(test_dir, "a", a, fmt="torch")
        write_tensor_to_disk(test_dir, "b", b, fmt="numpy")
        feature_data = [
            gb.OnDiskFeatureData(
                domain="node",
                type="paper",
                name="a",
                format="torch",
                path=os.path.join(test_dir, "a.pt"),
                in_memory=True,
            ),
            gb.OnDiskFeatureData(
                domain="edge",
                type="paper:cites:paper",
                name="b",
                format="numpy",
                path=os.path.join(test_dir, "b.npy"),
                in_memory=in_memory,
            ),
        ]
        feature_store = gb.TorchBasedFeatureStore(feature_data)

        expected_feature_store_str = (
            (
                "TorchBasedFeatureStore(\n"
                "    {(<OnDiskFeatureDataDomain.NODE: 'node'>, 'paper', 'a'): TorchBasedFeature(\n"
                "        feature=tensor([[1, 2, 4],\n"
                "                        [2, 5, 3]]),\n"
                "        metadata={},\n"
                "    ), (<OnDiskFeatureDataDomain.EDGE: 'edge'>, 'paper:cites:paper', 'b'): TorchBasedFeature(\n"
                "        feature=tensor([[[1, 2],\n"
                "                         [3, 4]],\n"
                "\n"
                "                        [[2, 5],\n"
                "                         [3, 4]]]),\n"
                "        metadata={},\n"
                "    )}\n"
                ")"
            )
            if in_memory
            else (
                "TorchBasedFeatureStore(\n"
                "    {(<OnDiskFeatureDataDomain.NODE: 'node'>, 'paper', 'a'): TorchBasedFeature(\n"
                "        feature=tensor([[1, 2, 4],\n"
                "                        [2, 5, 3]]),\n"
                "        metadata={},\n"
                "    ), (<OnDiskFeatureDataDomain.EDGE: 'edge'>, 'paper:cites:paper', 'b'): DiskBasedFeature(\n"
                "        feature=tensor([[[1, 2],\n"
                "                         [3, 4]],\n"
                "\n"
                "                        [[2, 5],\n"
                "                         [3, 4]]]),\n"
                "        metadata={},\n"
                "    )}\n"
                ")"
            )
        )

        assert repr(feature_store) == expected_feature_store_str, feature_store

        a = b = feature_data = None
        feature_store = expected_feature_store_str = None
