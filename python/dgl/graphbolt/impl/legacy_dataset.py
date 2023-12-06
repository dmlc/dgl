"""Graphbolt dataset for legacy DGLDataset."""
from typing import List, Union

import torch

from dgl.data import AsLinkPredDataset, AsNodePredDataset, DGLDataset
from ..base import etype_tuple_to_str
from ..dataset import Dataset, Task
from ..itemset import ItemSet, ItemSetDict
from ..sampling_graph import SamplingGraph
from .basic_feature_store import BasicFeatureStore
from .fused_csc_sampling_graph import from_dglgraph
from .ondisk_dataset import OnDiskTask
from .torch_based_feature_store import TorchBasedFeature


class LegacyDataset(Dataset):
    """A Graphbolt dataset for legacy DGLDataset."""

    def __init__(self, legacy: DGLDataset):
        self._init_as_homogeneous_link_pred(legacy)
        return
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

    def _init_as_heterogeneous_node_pred(self, legacy: DGLDataset):
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

            def _init_item_set_dict(idx, labels):
                item_set_dict = {}
                for key in idx.keys():
                    item_set = ItemSet(
                        (idx[key], labels[key][idx[key]]),
                        names=("seed_nodes", "labels"),
                    )
                    item_set_dict[key] = item_set
                return ItemSetDict(item_set_dict)

            train_set = _init_item_set_dict(split_idx["train"], labels)
            validation_set = _init_item_set_dict(split_idx["valid"], labels)
            test_set = _init_item_set_dict(split_idx["test"], labels)
            task = OnDiskTask(metadata, train_set, validation_set, test_set)
            tasks.append(task)
            self._tasks = tasks

            item_set_dict = {}
            for ntype in graph.ntypes:
                item_set = ItemSet(graph.num_nodes(ntype), names="seed_nodes")
                item_set_dict[ntype] = item_set
            self._all_nodes_set = ItemSetDict(item_set_dict)

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

    def _init_as_homogeneous_node_pred(self, legacy: DGLDataset):
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
            names=("seed_nodes", "labels"),
        )
        validation_set = ItemSet(
            (legacy.val_idx, validation_labels),
            names=("seed_nodes", "labels"),
        )
        test_set = ItemSet(
            (legacy.test_idx, test_labels), names=("seed_nodes", "labels")
        )
        task = OnDiskTask(metadata, train_set, validation_set, test_set)
        tasks.append(task)
        self._tasks = tasks

        num_nodes = legacy[0].num_nodes()
        self._all_nodes_set = ItemSet(num_nodes, names="seed_nodes")
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

    def _init_as_homogeneous_link_pred(self, legacy: DGLDataset):
        legacy = AsLinkPredDataset(legacy)

        # Initialize tasks.
        tasks = []
        metadata = {"name": "link_prediction"}
        graph = legacy.train_graph
        node_pairs = torch.stack(graph.edges()).T
        train_set = ItemSet(node_pairs, names="node_pairs")

        def _init_item_set_with_neg(edges):
            postive_edges, (negative_src, negative_dst) = edges
            node_pairs = torch.stack(postive_edges).T
            num_edges = node_pairs.size(dim=0)
            negative_src = negative_src.view(num_edges, -1)
            negative_dst = negative_dst.view(num_edges, -1)
            return ItemSet(
                (node_pairs, negative_src, negative_dst),
                names=("node_pairs", "negative_srcs", "negative_dsts"),
            )

        validation_set = _init_item_set_with_neg(legacy.val_edges)
        test_set = _init_item_set_with_neg(legacy.test_edges)
        task = OnDiskTask(metadata, train_set, validation_set, test_set)
        tasks.append(task)
        self._tasks = tasks

        num_nodes = graph.num_nodes()
        self._all_nodes_set = ItemSet(num_nodes, names="seed_nodes")
        features = {}
        for name in graph.ndata.keys():
            tensor = graph.ndata[name]
            if tensor.dim() == 1:
                tensor = tensor.view(-1, 1)
            features[("node", None, name)] = TorchBasedFeature(tensor)
        for name in graph.edata.keys():
            tensor = graph.edata[name]
            if tensor.dim() == 1:
                tensor = tensor.view(-1, 1)
            features[("edge", None, name)] = TorchBasedFeature(tensor)
        self._feature = BasicFeatureStore(features)
        self._graph = from_dglgraph(legacy.train_graph, is_homogeneous=True)
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
    def all_nodes_set(self) -> Union[ItemSet, ItemSetDict]:
        """Return the itemset containing all nodes."""
        return self._all_nodes_set
