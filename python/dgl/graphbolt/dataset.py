"""GraphBolt Dataset."""

from typing import List, Optional

import pydantic
import pydantic_yaml

from .feature_store import FeatureStore
from .itemset import ItemSet, ItemSetDict

__all__ = ["Dataset", "OnDiskDataset"]


class Dataset:
    """An abstract dataset.

    Dataset provides abstraction for accessing the data required for training.
    The data abstraction could be a native CPU memory block, a shared memory
    block, a file handle of an opened file on disk, a service that provides
    the API to access the data e.t.c. There are 3 primary components in the
    dataset: *Train-Validation-Test Set*, *Feature Storage*, *Graph Topology*.

    *Train-Validation-Test Set*:
    The training-validation-testing (TVT) set which is used to train the neural
    networks. We calculate the embeddings based on their respective features
    and the graph structure, and then utilize the embeddings to optimize the
    neural network parameters.

    *Feature Storage*:
    A key-value store which stores node/edge/graph features.

    *Graph Topology*:
    Graph topology is used by the subgraph sampling algorithm to
    generate a subgraph.
    """

    def train_set(self) -> ItemSet or ItemSetDict:
        """Return the training set."""
        raise NotImplementedError

    def validation_set(self) -> ItemSet or ItemSetDict:
        """Return the validation set."""
        raise NotImplementedError

    def test_set(self) -> ItemSet or ItemSetDict:
        """Return the test set."""
        raise NotImplementedError

    def graph(self) -> object:
        """Return the graph."""
        raise NotImplementedError

    def feature(self) -> FeatureStore:
        """Return the feature."""
        raise NotImplementedError


class OnDiskDataFormatEnum(pydantic_yaml.YamlStrEnum):
    """Enum of data format."""

    TORCH = "torch"
    NUMPY = "numpy"


class OnDiskTVTSet(pydantic.BaseModel):
    """Train-Validation-Test set."""

    type_name: str
    format: OnDiskDataFormatEnum
    path: str


class OnDiskMetaData(pydantic_yaml.YamlModel):
    """Metadata specification in YAML.

    As multiple node/edge types and multiple splits are supported, each TVT set
    is a list of list of ``OnDiskTVTSet``.
    """

    train_set: Optional[List[List[OnDiskTVTSet]]]
    validation_set: Optional[List[List[OnDiskTVTSet]]]
    test_set: Optional[List[List[OnDiskTVTSet]]]


class OnDiskDataset(Dataset):
    """An on-disk dataset.

    An on-disk dataset is a dataset which reads graph topology, feature data
    and TVT set from disk. Due to limited resources, the data which are too
    large to fit into RAM will remain on disk while others reside in RAM once
    ``OnDiskDataset`` is initialized. This behavior could be controled by user
    via ``in_memory`` field in YAML file.

    A full example of YAML file is as follows:

    .. code-block:: yaml

        train_set:
          - - type_name: paper
              format: numpy
              path: set/paper-train.npy
        validation_set:
          - - type_name: paper
              format: numpy
              path: set/paper-validation.npy
        test_set:
          - - type_name: paper
              format: numpy
              path: set/paper-test.npy

    Parameters
    ----------
    path: str
        The YAML file path.
    """

    def __init__(self, path: str) -> None:
        with open(path, "r") as f:
            self._meta = OnDiskMetaData.parse_raw(f.read(), proto="yaml")

    def train_set(self) -> ItemSet or ItemSetDict:
        """Return the training set."""
        raise NotImplementedError

    def validation_set(self) -> ItemSet or ItemSetDict:
        """Return the validation set."""
        raise NotImplementedError

    def test_set(self) -> ItemSet or ItemSetDict:
        """Return the test set."""
        raise NotImplementedError

    def graph(self) -> object:
        """Return the graph."""
        raise NotImplementedError

    def feature(self) -> FeatureStore:
        """Return the feature."""
        raise NotImplementedError
