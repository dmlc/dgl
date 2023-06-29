"""GraphBolt Dataset."""

from dgl import DGLGraph
from .feature_store import FeatureStore
from .graph_storage import CSCSamplingGraph
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
    Graph topology is eihter used by the subgraph sampling algorithm to
    generate a subgraph or used by model train directly.
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

    def graph(self) -> CSCSamplingGraph or DGLGraph:
        """Return the graph."""
        raise NotImplementedError

    def feature(self) -> FeatureStore:
        """Return the feature."""
        raise NotImplementedError
