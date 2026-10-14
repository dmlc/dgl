import backend as F

import dgl.graphbolt as gb
import pytest


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
