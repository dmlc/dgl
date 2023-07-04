import os
import tempfile

import pydantic
import pytest
from dgl import graphbolt as gb


def test_Dataset():
    dataset = gb.Dataset()
    with pytest.raises(NotImplementedError):
        _ = dataset.train_set()
    with pytest.raises(NotImplementedError):
        _ = dataset.validation_set()
    with pytest.raises(NotImplementedError):
        _ = dataset.test_set()
    with pytest.raises(NotImplementedError):
        _ = dataset.graph()
    with pytest.raises(NotImplementedError):
        _ = dataset.feature()


def test_OnDiskDataset_TVTSet():
    """Test OnDiskDataset with TVTSet."""
    with tempfile.TemporaryDirectory() as test_dir:
        yaml_content = """
        train_set:
          - - type_name: paper
              format: torch
              path: set/paper-train.pt
            - type_name: 'paper:cites:paper'
              format: numpy
              path: set/cites-train.pt
        """
        yaml_file = os.path.join(test_dir, "test.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)
        _ = gb.OnDiskDataset(yaml_file)

        # Invalid format.
        yaml_content = """
        train_set:
          - - type_name: paper
              format: torch_invalid
              path: set/paper-train.pt
            - type_name: 'paper:cites:paper'
              format: numpy_invalid
              path: set/cites-train.pt
        """
        with open(yaml_file, "w") as f:
            f.write(yaml_content)
        with pytest.raises(pydantic.ValidationError):
            _ = gb.OnDiskDataset(yaml_file)
