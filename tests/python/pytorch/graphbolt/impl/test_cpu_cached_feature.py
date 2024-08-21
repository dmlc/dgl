import os
import tempfile
import unittest

import backend as F
import numpy as np
import pytest
import torch

from dgl import graphbolt as gb


def to_on_disk_numpy(test_dir, name, t):
    path = os.path.join(test_dir, name + ".npy")
    np.save(path, t.numpy())
    return path


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ],
)
@pytest.mark.parametrize("policy", ["s3-fifo", "sieve", "lru", "clock"])
def test_cpu_cached_feature(dtype, policy):
    cache_size_a = 32
    cache_size_b = 64
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    b = torch.tensor([[[1, 2], [3, 4]], [[4, 5], [6, 7]]], dtype=dtype)

    pin_memory = F._default_context_str == "gpu"

    cache_size_a *= a[:1].nbytes
    cache_size_b *= b[:1].nbytes

    feat_store_a = gb.cpu_cached_feature(
        gb.TorchBasedFeature(a), cache_size_a, policy, pin_memory
    )
    feat_store_b = gb.cpu_cached_feature(
        gb.TorchBasedFeature(b), cache_size_b, policy, pin_memory
    )

    # Test read the entire feature.
    assert torch.equal(feat_store_a.read(), a)
    assert torch.equal(feat_store_b.read(), b)

    # Test read with ids.
    assert torch.equal(
        # Test read when ids are on a different device.
        feat_store_a.read(torch.tensor([0], device=F.ctx())),
        torch.tensor([[1, 2, 3]], dtype=dtype, device=F.ctx()),
    )
    assert torch.equal(
        feat_store_b.read(torch.tensor([1, 1])),
        torch.tensor([[[4, 5], [6, 7]], [[4, 5], [6, 7]]], dtype=dtype),
    )
    assert torch.equal(
        feat_store_a.read(torch.tensor([1, 1])),
        torch.tensor([[4, 5, 6], [4, 5, 6]], dtype=dtype),
    )
    assert torch.equal(
        feat_store_b.read(torch.tensor([0])),
        torch.tensor([[[1, 2], [3, 4]]], dtype=dtype),
    )
    # The cache should be full now for the large cache sizes, %100 hit expected.
    total_miss = feat_store_a._feature.total_miss
    feat_store_a.read(torch.tensor([0, 1]))
    assert total_miss == feat_store_a._feature.total_miss
    total_miss = feat_store_b._feature.total_miss
    feat_store_b.read(torch.tensor([0, 1]))
    assert total_miss == feat_store_b._feature.total_miss
    assert feat_store_a._feature.miss_rate == feat_store_a.miss_rate

    # Test get the size and count of the entire feature.
    assert feat_store_a.size() == torch.Size([3])
    assert feat_store_b.size() == torch.Size([2, 2])
    assert feat_store_a.count() == a.size(0)
    assert feat_store_b.count() == b.size(0)

    # Test update the entire feature.
    feat_store_a.update(torch.tensor([[0, 1, 2], [3, 5, 2]], dtype=dtype))
    assert torch.equal(
        feat_store_a.read(),
        torch.tensor([[0, 1, 2], [3, 5, 2]], dtype=dtype),
    )

    # Test update with ids.
    feat_store_a.update(
        torch.tensor([[2, 0, 1]], dtype=dtype),
        torch.tensor([0]),
    )
    assert torch.equal(
        feat_store_a.read(),
        torch.tensor([[2, 0, 1], [3, 5, 2]], dtype=dtype),
    )

    # Test with different dimensionality
    feat_store_a.update(b)
    assert torch.equal(feat_store_a.read(), b)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ],
)
def test_cpu_cached_feature_read_async(dtype):
    a = torch.randint(0, 2, [1000, 13], dtype=dtype)

    cache_size = 256 * a[:1].nbytes

    feat_store = gb.cpu_cached_feature(gb.TorchBasedFeature(a), cache_size)

    # Test read with ids.
    ids1 = torch.tensor([0, 15, 71, 101])
    ids2 = torch.tensor([71, 101, 202, 303])
    for ids in [ids1, ids2]:
        reader = feat_store.read_async(ids)
        for _ in range(feat_store.read_async_num_stages(ids.device)):
            values = next(reader)
        assert torch.equal(values.wait(), a[ids])


@unittest.skipIf(
    not torch.ops.graphbolt.detect_io_uring(),
    reason="DiskBasedFeature is not available on this system.",
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
    ],
)
def test_cpu_cached_disk_feature_read_async(dtype):
    a = torch.randint(0, 2, [1000, 13], dtype=dtype)

    cache_size = 256 * a[:1].nbytes

    ids1 = torch.tensor([0, 15, 71, 101])
    ids2 = torch.tensor([71, 101, 202, 303])

    with tempfile.TemporaryDirectory() as test_dir:
        path = to_on_disk_numpy(test_dir, "tensor", a)

        feat_store = gb.cpu_cached_feature(
            gb.DiskBasedFeature(path=path), cache_size
        )

        # Test read feature.
        for ids in [ids1, ids2]:
            reader = feat_store.read_async(ids)
            for _ in range(feat_store.read_async_num_stages(ids.device)):
                values = next(reader)
            assert torch.equal(values.wait(), a[ids])

        feat_store = None
