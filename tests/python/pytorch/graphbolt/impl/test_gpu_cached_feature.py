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
    np.save(path, t.cpu().numpy())
    return path


def _skip_condition_cached_feature():
    return (F._default_context_str != "gpu") or (
        torch.cuda.get_device_capability()[0] < 7
    )


def _reason_to_skip_cached_feature():
    if F._default_context_str != "gpu":
        return "GPUCachedFeature tests are available only when testing the GPU backend."

    return "GPUCachedFeature requires a Volta or later generation NVIDIA GPU."


@unittest.skipIf(
    _skip_condition_cached_feature(),
    reason=_reason_to_skip_cached_feature(),
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
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ],
)
@pytest.mark.parametrize("cache_size_a", [1, 1024])
@pytest.mark.parametrize("cache_size_b", [1, 1024])
def test_gpu_cached_feature(dtype, cache_size_a, cache_size_b):
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype, pin_memory=True)
    b = torch.tensor(
        [[[1, 2], [3, 4]], [[4, 5], [6, 7]]], dtype=dtype, pin_memory=True
    )

    cache_size_a *= a[:1].element_size() * a[:1].numel()
    cache_size_b *= b[:1].element_size() * b[:1].numel()

    feat_store_a = gb.gpu_cached_feature(gb.TorchBasedFeature(a), cache_size_a)
    feat_store_b = gb.gpu_cached_feature(gb.TorchBasedFeature(b), cache_size_b)

    # Test read the entire feature.
    assert torch.equal(feat_store_a.read(), a.to("cuda"))
    assert torch.equal(feat_store_b.read(), b.to("cuda"))

    # Test read with ids.
    assert torch.equal(
        feat_store_a.read(torch.tensor([0]).to("cuda")),
        torch.tensor([[1, 2, 3]], dtype=dtype).to("cuda"),
    )
    assert torch.equal(
        feat_store_b.read(torch.tensor([1, 1]).to("cuda")),
        torch.tensor([[[4, 5], [6, 7]], [[4, 5], [6, 7]]], dtype=dtype).to(
            "cuda"
        ),
    )
    assert torch.equal(
        feat_store_a.read(torch.tensor([1, 1]).to("cuda")),
        torch.tensor([[4, 5, 6], [4, 5, 6]], dtype=dtype).to("cuda"),
    )
    assert torch.equal(
        feat_store_b.read(torch.tensor([0]).to("cuda")),
        torch.tensor([[[1, 2], [3, 4]]], dtype=dtype).to("cuda"),
    )
    # The cache should be full now for the large cache sizes, %100 hit expected.
    if cache_size_a >= 1024:
        total_miss = feat_store_a._feature.total_miss
        feat_store_a.read(torch.tensor([0, 1]).to("cuda"))
        assert total_miss == feat_store_a._feature.total_miss
    if cache_size_b >= 1024:
        total_miss = feat_store_b._feature.total_miss
        feat_store_b.read(torch.tensor([0, 1]).to("cuda"))
        assert total_miss == feat_store_b._feature.total_miss
    assert feat_store_a._feature.miss_rate == feat_store_a.miss_rate

    # Test get the size and count of the entire feature.
    assert feat_store_a.size() == torch.Size([3])
    assert feat_store_b.size() == torch.Size([2, 2])
    assert feat_store_a.count() == a.size(0)
    assert feat_store_b.count() == b.size(0)

    # Test update the entire feature.
    feat_store_a.update(
        torch.tensor([[0, 1, 2], [3, 5, 2]], dtype=dtype).to("cuda")
    )
    assert torch.equal(
        feat_store_a.read(),
        torch.tensor([[0, 1, 2], [3, 5, 2]], dtype=dtype).to("cuda"),
    )

    # Test update with ids.
    feat_store_a.update(
        torch.tensor([[2, 0, 1]], dtype=dtype).to("cuda"),
        torch.tensor([0]).to("cuda"),
    )
    assert torch.equal(
        feat_store_a.read(),
        torch.tensor([[2, 0, 1], [3, 5, 2]], dtype=dtype).to("cuda"),
    )

    # Test with different dimensionality
    feat_store_a.update(b)
    assert torch.equal(feat_store_a.read(), b.to("cuda"))


@unittest.skipIf(
    _skip_condition_cached_feature(),
    reason=_reason_to_skip_cached_feature(),
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
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ],
)
@pytest.mark.parametrize("pin_memory", [False, True])
def test_gpu_cached_feature_read_async(dtype, pin_memory):
    a = torch.randint(0, 2, [1000, 13], dtype=dtype, pin_memory=pin_memory)
    a_cuda = a.to(F.ctx())

    cache_size = 256 * a[:1].nbytes

    feat_store = gb.gpu_cached_feature(gb.TorchBasedFeature(a), cache_size)

    # Test read with ids.
    ids1 = torch.tensor([0, 15, 71, 101], device=F.ctx())
    ids2 = torch.tensor([71, 101, 202, 303], device=F.ctx())
    for ids in [ids1, ids2]:
        reader = feat_store.read_async(ids)
        for _ in range(feat_store.read_async_num_stages(ids.device)):
            values = next(reader)
        assert torch.equal(values.wait(), a_cuda[ids])


@unittest.skipIf(
    _skip_condition_cached_feature(),
    reason=_reason_to_skip_cached_feature(),
)
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
def test_gpu_cached_nested_feature_async(dtype):
    a = torch.randint(0, 2, [1000, 13], dtype=dtype, device=F.ctx())

    cache_size = 256 * a[:1].nbytes

    ids1 = torch.tensor([0, 15, 71, 101], device=F.ctx())
    ids2 = torch.tensor([71, 101, 202, 303], device=F.ctx())

    with tempfile.TemporaryDirectory() as test_dir:
        path = to_on_disk_numpy(test_dir, "tensor", a)

        disk_store = gb.DiskBasedFeature(path=path)
        feat_store1 = gb.gpu_cached_feature(disk_store, cache_size)
        feat_store2 = gb.gpu_cached_feature(
            gb.cpu_cached_feature(disk_store, cache_size * 2), cache_size
        )
        feat_store3 = gb.gpu_cached_feature(
            gb.cpu_cached_feature(disk_store, cache_size * 2, pin_memory=True),
            cache_size,
        )

        # Test read feature.
        for feat_store in [feat_store1, feat_store2, feat_store3]:
            for ids in [ids1, ids2]:
                reader = feat_store.read_async(ids)
                for _ in range(feat_store.read_async_num_stages(ids.device)):
                    values = next(reader)
                assert torch.equal(values.wait(), a[ids])

        feat_store1 = feat_store2 = feat_store3 = disk_store = None
