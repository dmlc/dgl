import os
import tempfile

import numpy as np

import pydantic
import pytest
from dgl import graphbolt as gb


def test_Dataset():
    dataset = gb.Dataset()
    with pytest.raises(NotImplementedError):
        _ = dataset.train_sets()
    with pytest.raises(NotImplementedError):
        _ = dataset.validation_sets()
    with pytest.raises(NotImplementedError):
        _ = dataset.test_sets()
    with pytest.raises(NotImplementedError):
        _ = dataset.graph()
    with pytest.raises(NotImplementedError):
        _ = dataset.feature()
    with pytest.raises(NotImplementedError):
        _ = dataset.dataset_name
    with pytest.raises(NotImplementedError):
        _ = dataset.num_classes
    with pytest.raises(NotImplementedError):
        _ = dataset.num_labels
