"""Graphbolt dataset for legacy DGLDataset."""

from typing import List, Union

from ..base import etype_tuple_to_str
from ..dataset import Dataset, Task
from ..itemset import HeteroItemSet, ItemSet
from ..sampling_graph import SamplingGraph
from .basic_feature_store import BasicFeatureStore
from .fused_csc_sampling_graph import from_dglgraph
from .ondisk_dataset import OnDiskTask
from .torch_based_feature_store import TorchBasedFeature


class LegacyDataset(Dataset):
    """A Graphbolt dataset for legacy DGLDataset."""

    def __init__(self, legacy):
        # Only supports single graph cases.
        assert len(legacy) == 1
        graph = legacy[0]
        # Handle OGB Dataset.
        if isinstance(graph, tuple):
            graph, _ = graph
        if graph.is_homogeneous:
            self._init_as_homogeneous_node_pred(legacy)
        else:
            self._init_as_heterogeneous_node_pred(legacy)

    def _init_as_heterogeneous_node_pred(self, legacy):
        def _init_item_set_dict(idx, labels):
            item_set_dict = {}
            for key in idx.keys():
                item_set = ItemSet(
                    (idx[key], labels[key][idx[key]]),
                    names=("seeds", "labels"),
                )
                item_set_dict[key] = item_set
            return HeteroItemSet(item_set_dict)

        # OGB Dataset has the idx split.
        if hasattr(legacy, "get_idx_split"):
            graph, labels = legacy[0]
            split_idx = legacy.get_idx_split()

            # Initialize tasks.
            tasks = []
            metadata = {
                "num_classes": legacy.num_classes,
                "name": "node_classification",
            }
            train_set = _init_item_set_dict(split_idx["train"], labels)
            validation_set = _init_item_set_dict(split_idx["valid"], labels)
            test_set = _init_item_set_dict(split_idx["test"], labels)
            task = OnDiskTask(metadata, train_set, validation_set, test_set)
            tasks.append(task)
            self._tasks = tasks

            item_set_dict = {}
            for ntype in graph.ntypes:
                item_set = ItemSet(graph.num_nodes(ntype), names="seeds")
                item_set_dict[ntype] = item_set
            self._all_nodes_set = HeteroItemSet(item_set_dict)

            features = {}
            for ntype in graph.ntypes:
                for name in graph.nodes[ntype].data.keys():
                    tensor = graph.nodes[ntype].data[name]
                    if tensor.dim() == 1:
                        tensor = tensor.view(-1, 1)
                    features[("node", ntype, name)] = TorchBasedFeature(tensor)
            for etype in graph.canonical_etypes:
                for name in graph.edges[etype].data.keys():
                    tensor = graph.edges[etype].data[name]
                    if tensor.dim() == 1:
                        tensor = tensor.view(-1, 1)
                    gb_etype = etype_tuple_to_str(etype)
                    features[("edge", gb_etype, name)] = TorchBasedFeature(
                        tensor
                    )
            self._feature = BasicFeatureStore(features)
            self._graph = from_dglgraph(graph, is_homogeneous=False)
            self._dataset_name = legacy.name
        else:
            raise NotImplementedError(
                "Only support heterogeneous ogn node pred dataset"
            )

    def _init_as_homogeneous_node_pred(self, legacy):
        from dgl.data import AsNodePredDataset

        legacy = AsNodePredDataset(legacy)

        # Initialize tasks.
        tasks = []
        metadata = {
            "num_classes": legacy.num_classes,
            "name": "node_classification",
        }
        train_labels = legacy[0].ndata["label"][legacy.train_idx]
        validation_labels = legacy[0].ndata["label"][legacy.val_idx]
        test_labels = legacy[0].ndata["label"][legacy.test_idx]
        train_set = ItemSet(
            (legacy.train_idx, train_labels),
            names=("seeds", "labels"),
        )
        validation_set = ItemSet(
            (legacy.val_idx, validation_labels),
            names=("seeds", "labels"),
        )
        test_set = ItemSet(
            (legacy.test_idx, test_labels), names=("seeds", "labels")
        )
        task = OnDiskTask(metadata, train_set, validation_set, test_set)
        tasks.append(task)
        self._tasks = tasks

        num_nodes = legacy[0].num_nodes()
        self._all_nodes_set = ItemSet(num_nodes, names="seeds")
        features = {}
        for name in legacy[0].ndata.keys():
            tensor = legacy[0].ndata[name]
            if tensor.dim() == 1:
                tensor = tensor.view(-1, 1)
            features[("node", None, name)] = TorchBasedFeature(tensor)
        for name in legacy[0].edata.keys():
            tensor = legacy[0].edata[name]
            if tensor.dim() == 1:
                tensor = tensor.view(-1, 1)
            features[("edge", None, name)] = TorchBasedFeature(tensor)
        self._feature = BasicFeatureStore(features)
        self._graph = from_dglgraph(legacy[0], is_homogeneous=True)
        self._dataset_name = legacy.name

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
    def all_nodes_set(self) -> Union[ItemSet, HeteroItemSet]:
        """Return the itemset containing all nodes."""
        return self._all_nodes_set
