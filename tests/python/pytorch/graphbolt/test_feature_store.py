import torch
from dgl import graphbolt as gb


def test_feature_store():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([3, 4, 5])
    c = torch.tensor([[1, 2, 3], [4, 5, 6]])
    feature_store = gb.InMemoryFeatureStore({"a": a, "b": b, "c": c})
    assert torch.equal(feature_store.read_feature("a"), torch.tensor([1, 2, 3]))
    assert torch.equal(feature_store.read_feature("b"), torch.tensor([3, 4, 5]))
    assert torch.equal(
        feature_store.read_feature("a", torch.tensor([0, 2])),
        torch.tensor([1, 3]),
    )
    assert torch.equal(
        feature_store.read_feature("a", torch.tensor([1, 1])),
        torch.tensor([2, 2]),
    )
    assert torch.equal(
        feature_store.read_feature("c", torch.tensor([1])),
        torch.tensor([[4, 5, 6]]),
    )
    try:
        feature_store.read_feature("d")
        assert False
    except AssertionError:
        pass

    try:
        feature_store.read_feature("a", torch.tensor([0, 1, 2, 3]))
        assert False
    except IndexError:
        pass
