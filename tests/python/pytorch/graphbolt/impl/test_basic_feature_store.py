import pytest
import torch

from dgl import graphbolt as gb


def test_basic_feature_store():
    a = torch.tensor([3, 2, 1])
    b = torch.tensor([2, 5, 3])

    features = {}
    features[("node", "paper", "a")] = gb.TorchBasedFeature(a)
    features[("edge", "paper-cites-paper", "b")] = gb.TorchBasedFeature(b)

    feature_store = gb.BasicFeatureStore(features)
    assert torch.equal(
        feature_store.read("node", "paper", "a"), torch.tensor([3, 2, 1])
    )
    assert torch.equal(
        feature_store.read("edge", "paper-cites-paper", "b"),
        torch.tensor([2, 5, 3]),
    )
    assert torch.equal(
        feature_store.read("node", "paper", "a", torch.tensor([1, 2])),
        torch.tensor([3, 2]),
    )
