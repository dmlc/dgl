import unittest

import backend as F

import pytest
import torch

from dgl import graphbolt as gb


@unittest.skipIf(
    F._default_context_str != "gpu"
    or torch.cuda.get_device_capability()[0] < 7,
    reason="GPUCachedFeature requires a Volta or later generation NVIDIA GPU.",
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

    feat_store_a = gb.GPUCachedFeature(gb.TorchBasedFeature(a), cache_size_a)
    feat_store_b = gb.GPUCachedFeature(gb.TorchBasedFeature(b), cache_size_b)

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

    # Test get the size of the entire feature with ids.
    assert feat_store_a.size() == torch.Size([3])
    assert feat_store_b.size() == torch.Size([2, 2])

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
