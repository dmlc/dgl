"""GraphBolt Dataset."""

from typing import Dict, List, Union

from .feature_store import FeatureStore
from .itemset import HeteroItemSet, ItemSet
from .sampling_graph import SamplingGraph

__all__ = [
    "Task",
    "Dataset",
]


class Task:
    """An abstract task which consists of meta information and
    Train/Validation/Test Set.

    * meta information
        The meta information of a task includes any kinds of data that are
        defined by the user in YAML when instantiating the task.

    * Train/Validation/Test Set
        The train/validation/test (TVT) set which is used to train the neural
        networks. We calculate the embeddings based on their respective features
        and the graph structure, and then utilize the embeddings to optimize the
        neural network parameters.
    """

    @property
    def metadata(self) -> Dict:
        """Return the task metadata."""
        raise NotImplementedError

    @property
    def train_set(self) -> Union[ItemSet, HeteroItemSet]:
        """Return the training set."""
        raise NotImplementedError

    @property
    def validation_set(self) -> Union[ItemSet, HeteroItemSet]:
        """Return the validation set."""
        raise NotImplementedError

    @property
    def test_set(self) -> Union[ItemSet, HeteroItemSet]:
        """Return the test set."""
        raise NotImplementedError


class Dataset:
    """An abstract dataset which provides abstraction for accessing the data
    required for training.

    The data abstraction could be a native CPU memory block, a shared memory
    block, a file handle of an opened file on disk, a service that provides
    the API to access the data e.t.c. There are 3 primary components in the
    dataset:

    * Task
        A task consists of several meta information and the
        Train/Validation/Test Set. A dataset could have multiple tasks.

    * Feature Storage
        A key-value store which stores node/edge/graph features.

    * Graph Topology
        Graph topology is used by the subgraph sampling algorithm to generate
        a subgraph.
    """

    @property
    def tasks(self) -> List[Task]:
        """Return the tasks."""
        raise NotImplementedError

    @property
    def graph(self) -> SamplingGraph:
        """Return the graph."""
        raise NotImplementedError

    @property
    def feature(self) -> FeatureStore:
        """Return the feature."""
        raise NotImplementedError

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        raise NotImplementedError

    @property
    def all_nodes_set(self) -> Union[ItemSet, HeteroItemSet]:
        """Return the itemset containing all nodes."""
        raise NotImplementedError
