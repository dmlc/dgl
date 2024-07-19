import backend as F

import pytest
import torch

from dgl import graphbolt as gb


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

    feat_store_a = gb.CPUCachedFeature(
        gb.TorchBasedFeature(a), cache_size_a, policy, pin_memory
    )
    feat_store_b = gb.CPUCachedFeature(
        gb.TorchBasedFeature(b), cache_size_b, policy, pin_memory
    )

    # Test read the entire feature.
    assert torch.equal(feat_store_a.read(), a)
    assert torch.equal(feat_store_b.read(), b)

    # Test read with ids.
    assert torch.equal(
        feat_store_a.read(torch.tensor([0])),
        torch.tensor([[1, 2, 3]], dtype=dtype),
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

    # Test get the size of the entire feature with ids.
    assert feat_store_a.size() == torch.Size([3])
    assert feat_store_b.size() == torch.Size([2, 2])

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
