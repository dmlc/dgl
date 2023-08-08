"""GraphBolt Dataset."""
from enum import Enum
from typing import List, Optional

from .feature_store import FeatureStore
from .itemset import ItemSet, ItemSetDict

__all__ = [
    "GNNTaskTypes",
    "TaskMetadata",
    "ClassificationTaskMetadata",
    "Task",
    "Dataset",
]


class GNNTaskTypes(str, Enum):
    """Enum of task names."""

    NODE_CLASSIFICATION = "node_classification"
    NODE_REGRESSION = "node_regression"
    EDGE_CLASSIFICATION = "edge_classification"
    EDGE_REGRESSION = "edge_regression"
    LINK_PREDICTION = "link_prediction"
    GRAPH_CLASSIFICATION = "graph_classification"


class TaskMetadata:
    """Task metadata."""

    def __init__(self, task_type: str):
        """Initialize a task metadata.

        Parameters
        ----------
        task_type : GNNTaskTypes
            Task type.
        """
        self._task_type = task_type

    @property
    def task_type(self) -> str:
        """Return the task type."""
        return self._task_type.value


class ClassificationTaskMetadata(TaskMetadata):
    """Classification task metadata."""

    def __init__(
        self,
        task_type: GNNTaskTypes,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
    ):
        """Initialize a classification task metadata.

        Parameters
        ----------
        task_type : GNNTaskTypes
            Task type.
        num_classes : int
            Number of classes.
        num_labels : int
            Number of labels.
        """
        super().__init__(task_type)
        self._num_classes = num_classes
        self._num_labels = num_labels

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return self._num_classes

    @property
    def num_labels(self) -> int:
        """Return the number of labels."""
        return self._num_labels


class Task:
    """An task.

    Task consists of several meta information and the *Train-Validation-Test Set*.

    *Train-Validation-Test Set*:
    The training-validation-testing (TVT) set which is used to train the neural
    networks. We calculate the embeddings based on their respective features
    and the graph structure, and then utilize the embeddings to optimize the
    neural network parameters.
    """

    def __init__(
        self,
        metadata: TaskMetadata,
        train_set: ItemSet or ItemSetDict,
        validation_set: ItemSet or ItemSetDict,
        test_set: ItemSet or ItemSetDict,
    ):
        """Initialize a task.

        Parameters
        ----------
        metadata : TaskMetadata
            Metadata.
        train_set : ItemSet or ItemSetDict
            Training set.
        validation_set : ItemSet or ItemSetDict
            Validation set.
        test_set : ItemSet or ItemSetDict
            Test set.
        """
        self._metadata = metadata
        self._train_set = train_set
        self._validation_set = validation_set
        self._test_set = test_set

    @property
    def metadata(self) -> TaskMetadata:
        """Return the task metadata."""
        return self._metadata

    @property
    def train_set(self) -> ItemSet or ItemSetDict:
        """Return the training set."""
        return self._train_set

    @property
    def validation_set(self) -> ItemSet or ItemSetDict:
        """Return the validation set."""
        return self._validation_set

    @property
    def test_set(self) -> ItemSet or ItemSetDict:
        """Return the test set."""
        return self._test_set


class Dataset:
    """An abstract dataset.

    Dataset provides abstraction for accessing the data required for training.
    The data abstraction could be a native CPU memory block, a shared memory
    block, a file handle of an opened file on disk, a service that provides
    the API to access the data e.t.c. There are 3 primary components in the
    dataset: *Task*, *Feature Storage*, *Graph Topology*.

    *Task*:
    A task consists of several meta information and the
    *Train-Validation-Test Set*. A dataset could have multiple tasks.

    *Feature Storage*:
    A key-value store which stores node/edge/graph features.

    *Graph Topology*:
    Graph topology is used by the subgraph sampling algorithm to
    generate a subgraph.
    """

    @property
    def tasks(self) -> List[Task]:
        """Return the tasks."""
        raise NotImplementedError

    @property
    def graph(self) -> object:
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
