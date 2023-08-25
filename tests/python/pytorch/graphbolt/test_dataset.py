import pytest

from dgl import graphbolt as gb


def test_Dataset():
    dataset = gb.Dataset()
    with pytest.raises(NotImplementedError):
        _ = dataset.tasks
    with pytest.raises(NotImplementedError):
        _ = dataset.graph
    with pytest.raises(NotImplementedError):
        _ = dataset.feature
    with pytest.raises(NotImplementedError):
        _ = dataset.dataset_name
