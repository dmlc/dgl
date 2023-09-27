import pytest
import torch

from dgl import graphbolt as gb


def test_basic_feature_store_homo():
    a = torch.tensor([[1, 2, 4], [2, 5, 3]])
    b = torch.tensor([[[1, 2], [3, 4]], [[2, 5], [4, 3]]])

    features = {}
    features[("node", None, "a")] = gb.TorchBasedFeature(a)
    features[("node", None, "b")] = gb.TorchBasedFeature(b)

    feature_store = gb.BasicFeatureStore(features)

    # Test read the entire feature.
    assert torch.equal(
        feature_store.read("node", None, "a"),
        torch.tensor([[1, 2, 4], [2, 5, 3]]),
    )
    assert torch.equal(
        feature_store.read("node", None, "b"),
        torch.tensor([[[1, 2], [3, 4]], [[2, 5], [4, 3]]]),
    )

    # Test read with ids.
    assert torch.equal(
        feature_store.read("node", None, "a", torch.tensor([0])),
        torch.tensor([[1, 2, 4]]),
    )
    assert torch.equal(
        feature_store.read("node", None, "b", torch.tensor([0])),
        torch.tensor([[[1, 2], [3, 4]]]),
    )


def test_basic_feature_store_hetero():
    a = torch.tensor([[1, 2, 4], [2, 5, 3]])
    b = torch.tensor([[[6], [8]], [[8], [9]]])

    features = {}
    features[("node", "author", "a")] = gb.TorchBasedFeature(a)
    features[("edge", "paper:cites:paper", "b")] = gb.TorchBasedFeature(b)

    feature_store = gb.BasicFeatureStore(features)

    # Test read the entire feature.
    assert torch.equal(
        feature_store.read("node", "author", "a"),
        torch.tensor([[1, 2, 4], [2, 5, 3]]),
    )
    assert torch.equal(
        feature_store.read("edge", "paper:cites:paper", "b"),
        torch.tensor([[[6], [8]], [[8], [9]]]),
    )

    # Test read with ids.
    assert torch.equal(
        feature_store.read("node", "author", "a", torch.tensor([0])),
        torch.tensor([[1, 2, 4]]),
    )


def test_basic_feature_store_errors():
    a = torch.tensor([3, 2, 1])
    b = torch.tensor([[1, 2, 4], [2, 5, 3]])

    features = {}
    # Test error when dimension of the value is illegal.
    with pytest.raises(
        AssertionError,
        match=rf"The dimension of the value is illegal."
    ):
        features[("node", "paper", "a")] = gb.TorchBasedFeature(a)
    features[("node", "author", "b")] = gb.TorchBasedFeature(b)

    feature_store = gb.BasicFeatureStore(features)

    # Test error when key does not exist.
    with pytest.raises(KeyError):
        feature_store.read("node", "paper", "b")

    # Test error when at least one id is out of bound.
    with pytest.raises(IndexError):
        feature_store.read("node", "author", "b", torch.tensor([0, 3]))
