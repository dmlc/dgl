import unittest

import backend as F

import pytest
import torch

from dgl import graphbolt as gb


@unittest.skipIf(
    F._default_context_str != "gpu",
    reason="GPUCachedFeature requires a GPU.",
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
def test_gpu_cached_feature(dtype):
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype, pin_memory=True)
    b = torch.tensor(
        [[[1, 2], [3, 4]], [[4, 5], [6, 7]]], dtype=dtype, pin_memory=True
    )

    feat_store_a = gb.GPUCachedFeature(gb.TorchBasedFeature(a), 2)
    feat_store_b = gb.GPUCachedFeature(gb.TorchBasedFeature(b), 1)

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
