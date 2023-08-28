import unittest

import backend as F

import torch

from dgl import graphbolt as gb


@unittest.skipIf(
    F._default_context_str != "gpu",
    reason="GPUCachedFeature requires a GPU.",
)
def test_gpu_cached_feature():
    a = torch.tensor([1, 2, 3]).to("cuda").float()
    b = torch.tensor([[1, 2, 3], [4, 5, 6]]).to("cuda").float()

    feat_store_a = gb.GPUCachedFeature(gb.TorchBasedFeature(a), 2)
    feat_store_b = gb.GPUCachedFeature(gb.TorchBasedFeature(b), 1)

    # Test read the entire feature.
    assert torch.equal(feat_store_a.read(), a)
    assert torch.equal(feat_store_b.read(), b)

    # Test read with ids.
    assert torch.equal(
        feat_store_a.read(torch.tensor([0, 2]).to("cuda")),
        torch.tensor([1.0, 3.0]).to("cuda"),
    )
    assert torch.equal(
        feat_store_a.read(torch.tensor([1, 1]).to("cuda")),
        torch.tensor([2.0, 2.0]).to("cuda"),
    )
    assert torch.equal(
        feat_store_b.read(torch.tensor([1]).to("cuda")),
        torch.tensor([[4.0, 5.0, 6.0]]).to("cuda"),
    )

    # Test update the entire feature.
    feat_store_a.update(torch.tensor([0.0, 1.0, 2.0]).to("cuda"))
    assert torch.equal(
        feat_store_a.read(), torch.tensor([0.0, 1.0, 2.0]).to("cuda")
    )

    # Test update with ids.
    feat_store_a.update(
        torch.tensor([2.0, 0.0]).to("cuda"), torch.tensor([0, 2]).to("cuda")
    )
    assert torch.equal(
        feat_store_a.read(), torch.tensor([2.0, 1.0, 0.0]).to("cuda")
    )
