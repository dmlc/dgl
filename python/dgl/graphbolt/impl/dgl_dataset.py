"""Graphbolt dataset from DGLDataset."""
from typing import Dict, List, Union

from dgl.data import AsNodePredDataset
from ..dataset import Dataset, Task
from ..itemset import ItemSet, ItemSetDict
from ..sampling_graph import SamplingGraph
from .basic_feature_store import BasicFeatureStore
from .fused_csc_sampling_graph import from_dglgraph, FusedCSCSamplingGraph
from .torch_based_feature_store import TorchBasedFeature


class DGLGraphboltTask(Task):
    def __init__(self, dgl_dataset: AsNodePredDataset):
        train_labels = dgl_dataset[0].ndata["label"][dgl_dataset.train_idx]
        validation_labels = dgl_dataset[0].ndata["label"][dgl_dataset.val_idx]
        test_labels = dgl_dataset[0].ndata["label"][dgl_dataset.test_idx]
        self._train_set = ItemSet(
            (dgl_dataset.train_idx, train_labels),
            names=("seed_nodes", "labels"),
        )
        self._validation_set = ItemSet(
            (dgl_dataset.val_idx, validation_labels),
            names=("seed_nodes", "labels"),
        )
        self._test_set = ItemSet(
            (dgl_dataset.test_idx, test_labels), names=("seed_nodes", "labels")
        )
        self._metadata = {"num_classes": dgl_dataset.num_classes}

    @property
    def metadata(self) -> Dict:
        """Return the task metadata."""
        return self._metadata

    @property
    def train_set(self) -> Union[ItemSet, ItemSetDict]:
        """Return the training set."""
        return self._train_set

    @property
    def validation_set(self) -> Union[ItemSet, ItemSetDict]:
        """Return the validation set."""
        return self._validation_set

    @property
    def test_set(self) -> Union[ItemSet, ItemSetDict]:
        """Return the test set."""
        return self._test_set


class DGLGraphboltDataset(Dataset):
    def __init__(self, dgl_dataset: AsNodePredDataset):
        assert len(dgl_dataset) == 1
        tasks = []
        tasks.append(DGLGraphboltTask(dgl_dataset))
        self._tasks = tasks
        num_nodes = dgl_dataset[0].num_nodes
        self._all_nodes_set = ItemSet(num_nodes, names="seed_nodes")
        features = {}
        for name in dgl_dataset[0].ndata.keys():
            tensor = dgl_dataset[0].ndata[name]
            if tensor.dim() == 1:
                tensor = tensor.view(-1, 1)
            features[("node", None, name)] = TorchBasedFeature(tensor)
        self._feature = BasicFeatureStore(features)
        self._graph = from_dglgraph(dgl_dataset[0], is_homogeneous=True)
        self._dataset_name = ""

    @property
    def tasks(self) -> List[Task]:
        """Return the tasks."""
        return self._tasks

    @property
    def graph(self) -> SamplingGraph:
        """Return the graph."""
        return self._graph

    @property
    def feature(self) -> BasicFeatureStore:
        """Return the feature."""
        return self._feature

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return self._dataset_name

    @property
    def all_nodes_set(self) -> Union[ItemSet, ItemSetDict]:
        """Return the itemset containing all nodes."""
        return self._all_nodes_set
