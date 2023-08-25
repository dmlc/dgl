import pytest
import torch

from dgl import graphbolt as gb


def test_basic_feature_store_homo():
    a = torch.tensor([3, 2, 1])
    b = torch.tensor([2, 5, 3])

    features = {}
    features[("node", None, "a")] = gb.TorchBasedFeature(a)
    features[("node", None, "b")] = gb.TorchBasedFeature(b)

    feature_store = gb.BasicFeatureStore(features)

    # Test read the entire feature.
    assert torch.equal(
        feature_store.read("node", None, "a"), torch.tensor([3, 2, 1])
    )
    assert torch.equal(
        feature_store.read("node", None, "b"), torch.tensor([2, 5, 3])
    )

    # Test read with ids.
    assert torch.equal(
        feature_store.read("node", None, "a", torch.tensor([0, 1])),
        torch.tensor([3, 2]),
    )


def test_basic_feature_store_hetero():
    a = torch.tensor([3, 2, 1])
    b = torch.tensor([2, 5, 3])
    c = torch.tensor([6, 8, 9])

    features = {}
    features[("node", "paper", "a")] = gb.TorchBasedFeature(a)
    features[("node", "author", "b")] = gb.TorchBasedFeature(b)
    features[("edge", "paper:cites:paper", "c")] = gb.TorchBasedFeature(c)

    feature_store = gb.BasicFeatureStore(features)

    # Test read the entire feature.
    assert torch.equal(
        feature_store.read("node", "paper", "a"), torch.tensor([3, 2, 1])
    )
    assert torch.equal(
        feature_store.read("node", "paper", "b"), torch.tensor([2, 5, 3])
    )
    assert torch.equal(
        feature_store.read("edge", "paper:cites:paper", "c"),
        torch.tensor([6, 8, 9]),
    )

    # Test read with ids.
    assert torch.equal(
        feature_store.read("node", "paper", "a", torch.tensor([0, 1])),
        torch.tensor([3, 2]),
    )
