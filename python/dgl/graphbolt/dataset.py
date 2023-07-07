"""GraphBolt Dataset."""

from typing import List, Optional

import pydantic
import pydantic_yaml

from .feature_store import FeatureStore
from .itemset import ItemSet, ItemSetDict
from .utils import read_data, tensor_to_tuple

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

    def train_sets(self) -> List[ItemSet] or List[ItemSetDict]:
        """Return the training sets."""
        raise NotImplementedError

    def validation_sets(self) -> List[ItemSet] or List[ItemSetDict]:
        """Return the validation sets."""
        raise NotImplementedError

    def test_sets(self) -> List[ItemSet] or List[ItemSetDict]:
        """Return the test sets."""
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

    type_name: Optional[str]
    format: OnDiskDataFormatEnum
    in_memory: Optional[bool] = True
    path: str


class OnDiskMetaData(pydantic_yaml.YamlModel):
    """Metadata specification in YAML.

    As multiple node/edge types and multiple splits are supported, each TVT set
    is a list of list of ``OnDiskTVTSet``.
    """

    train_sets: Optional[List[List[OnDiskTVTSet]]]
    validation_sets: Optional[List[List[OnDiskTVTSet]]]
    test_sets: Optional[List[List[OnDiskTVTSet]]]


class OnDiskDataset(Dataset):
    """An on-disk dataset.

    An on-disk dataset is a dataset which reads graph topology, feature data
    and TVT set from disk. Due to limited resources, the data which are too
    large to fit into RAM will remain on disk while others reside in RAM once
    ``OnDiskDataset`` is initialized. This behavior could be controled by user
    via ``in_memory`` field in YAML file.

    A full example of YAML file is as follows:

    .. code-block:: yaml

        train_sets:
          - - type_name: paper # could be null for homogeneous graph.
              format: numpy
              in_memory: true # If not specified, default to true.
              path: set/paper-train.npy
        validation_sets:
          - - type_name: paper
              format: numpy
              in_memory: true
              path: set/paper-validation.npy
        test_sets:
          - - type_name: paper
              format: numpy
              in_memory: true
              path: set/paper-test.npy

    Parameters
    ----------
    path: str
        The YAML file path.
    """

    def __init__(self, path: str) -> None:
        with open(path, "r") as f:
            self._meta = OnDiskMetaData.parse_raw(f.read(), proto="yaml")
        self._train_sets = self._init_tvt_sets(self._meta.train_sets)
        self._validation_sets = self._init_tvt_sets(self._meta.validation_sets)
        self._test_sets = self._init_tvt_sets(self._meta.test_sets)

    def train_sets(self) -> List[ItemSet] or List[ItemSetDict]:
        """Return the training set."""
        return self._train_sets

    def validation_sets(self) -> List[ItemSet] or List[ItemSetDict]:
        """Return the validation set."""
        return self._validation_sets

    def test_sets(self) -> List[ItemSet] or List[ItemSetDict]:
        """Return the test set."""
        return self._test_sets

    def graph(self) -> object:
        """Return the graph."""
        raise NotImplementedError

    def feature(self) -> FeatureStore:
        """Return the feature."""
        raise NotImplementedError

    def _init_tvt_sets(
        self, tvt_sets: List[List[OnDiskTVTSet]]
    ) -> List[ItemSet] or List[ItemSetDict]:
        """Initialize the TVT sets."""
        if (tvt_sets is None) or (len(tvt_sets) == 0):
            return None
        ret = []
        for tvt_set in tvt_sets:
            if (tvt_set is None) or (len(tvt_set) == 0):
                ret.append(None)
            if tvt_set[0].type_name is None:
                assert (
                    len(tvt_set) == 1
                ), "Only one TVT set is allowed if type_name is not specified."
                data = read_data(
                    tvt_set[0].path, tvt_set[0].format, tvt_set[0].in_memory
                )
                ret.append(ItemSet(tensor_to_tuple(data)))
            else:
                data = {}
                for tvt in tvt_set:
                    data[tvt.type_name] = ItemSet(
                        tensor_to_tuple(
                            read_data(tvt.path, tvt.format, tvt.in_memory)
                        )
                    )
                ret.append(ItemSetDict(data))
        return ret
