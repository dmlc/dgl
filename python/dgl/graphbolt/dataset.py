"""GraphBolt Dataset."""

from typing import Dict, List

from .feature_store import FeatureStore
from .itemset import ItemSet, ItemSetDict

__all__ = ["Dataset"]


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

    @property
    def train_sets(self) -> List[ItemSet] or List[ItemSetDict]:
        """Return the training sets."""
        raise NotImplementedError

    @property
    def validation_sets(self) -> List[ItemSet] or List[ItemSetDict]:
        """Return the validation sets."""
        raise NotImplementedError

    @property
    def test_sets(self) -> List[ItemSet] or List[ItemSetDict]:
        """Return the test sets."""
        raise NotImplementedError

    @property
    def graph(self) -> object:
        """Return the graph."""
        raise NotImplementedError

    @property
    def feature(self) -> Dict[object, FeatureStore]:
        """Return the feature."""
        raise NotImplementedError

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        raise NotImplementedError

    @property
    def num_labels(self) -> int:
        """Return the number of labels."""
        raise NotImplementedError
