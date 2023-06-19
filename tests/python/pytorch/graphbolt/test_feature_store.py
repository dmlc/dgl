import pytest
import torch
from dgl import graphbolt as gb


def test_in_memory_feature_store():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([3, 4, 5])
    c = torch.tensor([[1, 2, 3], [4, 5, 6]])
    feature_store = gb.InMemoryFeatureStore({"a": a, "b": b, "c": c})
    assert torch.equal(feature_store.read("a"), torch.tensor([1, 2, 3]))
    assert torch.equal(feature_store.read("b"), torch.tensor([3, 4, 5]))
    assert torch.equal(
        feature_store.read("a", torch.tensor([0, 2])),
        torch.tensor([1, 3]),
    )
    assert torch.equal(
        feature_store.read("a", torch.tensor([1, 1])),
        torch.tensor([2, 2]),
    )
    assert torch.equal(
        feature_store.read("c", torch.tensor([1])),
        torch.tensor([[4, 5, 6]]),
    )
    feature_store.update("a", torch.tensor([0, 1, 2]))
    assert torch.equal(feature_store.read("a"), torch.tensor([0, 1, 2]))
    assert torch.equal(
        feature_store.read("a", torch.tensor([0, 2])),
        torch.tensor([0, 2]),
    )
    with pytest.raises(AssertionError):
        feature_store.read("d")

    with pytest.raises(IndexError):
        feature_store.read("a", torch.tensor([0, 1, 2, 3]))
