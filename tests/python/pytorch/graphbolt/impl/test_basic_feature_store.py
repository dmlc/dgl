import pytest
import torch

from dgl import graphbolt as gb


def test_basic_feature_store_homo():
    a = torch.tensor([[1, 2, 4], [2, 5, 3]])
    b = torch.tensor([[[1, 2], [3, 4]], [[2, 5], [4, 3]]])
    metadata = {"max_value": 3}

    features = {}
    features[("node", None, "a")] = gb.TorchBasedFeature(a, metadata=metadata)
    features[("node", None, "b")] = gb.TorchBasedFeature(b)

    feature_store = gb.BasicFeatureStore(features)

    # Test __getitem__ to access the stored Feature.
    feature = feature_store[("node", None, "a")]
    assert isinstance(feature, gb.Feature)
    assert torch.equal(
        feature.read(),
        torch.tensor([[1, 2, 4], [2, 5, 3]]),
    )

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

    # Test get the size and count of the entire feature.
    assert feature_store.size("node", None, "a") == torch.Size([3])
    assert feature_store.size("node", None, "b") == torch.Size([2, 2])
    assert feature_store.count("node", None, "a") == a.size(0)
    assert feature_store.count("node", None, "b") == b.size(0)

    # Test get metadata of the feature.
    assert feature_store.metadata("node", None, "a") == metadata
    assert feature_store.metadata("node", None, "b") == {}

    # Test __setitem__ and __contains__ of FeatureStore.
    assert ("node", None, "c") not in feature_store
    feature_store[("node", None, "c")] = feature_store[("node", None, "a")]
    assert ("node", None, "c") in feature_store

    # Test get keys of the features.
    assert feature_store.keys() == [
        ("node", None, "a"),
        ("node", None, "b"),
        ("node", None, "c"),
    ]


def test_basic_feature_store_hetero():
    a = torch.tensor([[1, 2, 4], [2, 5, 3]])
    b = torch.tensor([[[6], [8]], [[8], [9]]])
    metadata = {"max_value": 3}

    features = {}
    features[("node", "author", "a")] = gb.TorchBasedFeature(
        a, metadata=metadata
    )
    features[("edge", "paper:cites", "b")] = gb.TorchBasedFeature(b)

    feature_store = gb.BasicFeatureStore(features)

    # Test __getitem__ to access the stored Feature.
    feature = feature_store[("node", "author", "a")]
    assert isinstance(feature, gb.Feature)
    assert torch.equal(
        feature.read(),
        torch.tensor([[1, 2, 4], [2, 5, 3]]),
    )

    # Test read the entire feature.
    assert torch.equal(
        feature_store.read("node", "author", "a"),
        torch.tensor([[1, 2, 4], [2, 5, 3]]),
    )
    assert torch.equal(
        feature_store.read("edge", "paper:cites", "b"),
        torch.tensor([[[6], [8]], [[8], [9]]]),
    )

    # Test read with ids.
    assert torch.equal(
        feature_store.read("node", "author", "a", torch.tensor([0])),
        torch.tensor([[1, 2, 4]]),
    )

    # Test get the size of the entire feature.
    assert feature_store.size("node", "author", "a") == torch.Size([3])
    assert feature_store.size("edge", "paper:cites", "b") == torch.Size([2, 1])

    # Test get metadata of the feature.
    assert feature_store.metadata("node", "author", "a") == metadata
    assert feature_store.metadata("edge", "paper:cites", "b") == {}

    # Test __setitem__ and __contains__ of FeatureStore.
    assert ("node", "author", "c") not in feature_store
    feature_store[("node", "author", "c")] = feature_store[
        ("node", "author", "a")
    ]
    assert ("node", "author", "c") in feature_store

    # Test get keys of the features.
    assert feature_store.keys() == [
        ("node", "author", "a"),
        ("edge", "paper:cites", "b"),
        ("node", "author", "c"),
    ]


def test_basic_feature_store_errors():
    a = torch.tensor([3, 2, 1])
    b = torch.tensor([[1, 2, 4], [2, 5, 3]])

    features = {}
    # Test error when dimension of the value is illegal.
    with pytest.raises(
        AssertionError,
        match=rf"dimension of torch_feature in TorchBasedFeature must be "
        rf"greater than 1, but got {a.dim()} dimension.",
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
