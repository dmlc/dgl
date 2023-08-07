import pytest
from dgl import graphbolt as gb


def test_Dataset():
    dataset = gb.Dataset()
    with pytest.raises(NotImplementedError):
        _ = dataset.train_set
    with pytest.raises(NotImplementedError):
        _ = dataset.validation_set
    with pytest.raises(NotImplementedError):
        _ = dataset.test_set
    with pytest.raises(NotImplementedError):
        _ = dataset.graph
    with pytest.raises(NotImplementedError):
        _ = dataset.feature
    with pytest.raises(NotImplementedError):
        _ = dataset.dataset_name
    with pytest.raises(NotImplementedError):
        _ = dataset.num_classes
    with pytest.raises(NotImplementedError):
        _ = dataset.num_labels
