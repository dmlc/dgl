import pytest
import torch

from dgl import graphbolt as gb


def test_basic_feature_store_homo():
    a = torch.tensor([3, 2, 1])
    b = torch.tensor([2, 5, 3])
    c = torch.tensor([[1, 2, 3], [4, 5, 6]])

    features = {}
    features[("node", None, "a")] = gb.TorchBasedFeature(a)
    features[("node", None, "b")] = gb.TorchBasedFeature(b)
    features[("node", None, "c")] = gb.TorchBasedFeature(c)

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

    # Test get the size of the entire feature.
    assert feature_store.size("node", None, "a") == torch.Size([1])
    assert feature_store.size("node", None, "b") == torch.Size([1])
    assert feature_store.size("node", None, "c") == torch.Size([3])


def test_basic_feature_store_hetero():
    a = torch.tensor([3, 2, 1])
    b = torch.tensor([2, 5, 3])
    c = torch.tensor([6, 8, 9])
    d = torch.tensor([[1, 2], [4, 5]])

    features = {}
    features[("node", "paper", "a")] = gb.TorchBasedFeature(a)
    features[("node", "author", "b")] = gb.TorchBasedFeature(b)
    features[("edge", "paper:cites:paper", "c")] = gb.TorchBasedFeature(c)
    features[("edge", "name:author", "d")] = gb.TorchBasedFeature(d)

    feature_store = gb.BasicFeatureStore(features)

    # Test read the entire feature.
    assert torch.equal(
        feature_store.read("node", "paper", "a"), torch.tensor([3, 2, 1])
    )
    assert torch.equal(
        feature_store.read("node", "author", "b"), torch.tensor([2, 5, 3])
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

    # Test get the size of the entire feature.
    assert feature_store.size("node", "paper", "a") == torch.Size([1])
    assert feature_store.size("node", "author", "b") == torch.Size([1])
    assert feature_store.size("edge", "name:author", "d") == torch.Size([2])


def test_basic_feature_store_errors():
    a = torch.tensor([3, 2, 1])
    b = torch.tensor([2, 5, 3])

    features = {}
    features[("node", "paper", "a")] = gb.TorchBasedFeature(a)
    features[("node", "author", "b")] = gb.TorchBasedFeature(b)

    feature_store = gb.BasicFeatureStore(features)

    # Test error when key does not exist.
    with pytest.raises(KeyError):
        feature_store.read("node", "paper", "b")

    # Test error when at least one id is out of bound.
    with pytest.raises(IndexError):
        feature_store.read("node", "paper", "a", torch.tensor([0, 3]))
