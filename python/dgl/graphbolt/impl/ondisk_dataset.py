"""GraphBolt OnDiskDataset."""

import os
import shutil

from copy import deepcopy
from typing import Dict, List

import pandas as pd
import torch
import yaml

import dgl

from ..dataset import Dataset, Task
from ..itemset import ItemSet, ItemSetDict
from ..utils import read_data, save_data

from .csc_sampling_graph import (
    CSCSamplingGraph,
    from_dglgraph,
    load_csc_sampling_graph,
    save_csc_sampling_graph,
)
from .ondisk_metadata import (
    OnDiskGraphTopology,
    OnDiskMetaData,
    OnDiskTaskData,
    OnDiskTVTSet,
)
from .torch_based_feature_store import TorchBasedFeatureStore

__all__ = ["OnDiskDataset", "preprocess_ondisk_dataset"]


def _copy_or_convert_data(
    input_path,
    output_path,
    input_format,
    output_format="numpy",
    in_memory=True,
):
    """Copy or convert the data from input_path to output_path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if input_format == "numpy":
        # If the original format is numpy, just copy the file.
        shutil.copyfile(input_path, output_path)
    else:
        # If the original format is not numpy, convert it to numpy.
        data = read_data(input_path, input_format, in_memory)
        save_data(data, output_path, output_format)


def preprocess_ondisk_dataset(dataset_dir: str) -> str:
    """Preprocess the on-disk dataset. Parse the input config file,
    load the data, and save the data in the format that GraphBolt supports.

    Parameters
    ----------
    dataset_dir : str
        The path to the dataset directory.

    Returns
    -------
    output_config_path : str
        The path to the output config file.
    """
    # Check if the dataset path is valid.
    if not os.path.exists(dataset_dir):
        raise RuntimeError(f"Invalid dataset path: {dataset_dir}")

    # Check if the dataset_dir is a directory.
    if not os.path.isdir(dataset_dir):
        raise RuntimeError(
            f"The dataset must be a directory. But got {dataset_dir}"
        )

    # 0. Check if the dataset is already preprocessed.
    preprocess_metadata_path = os.path.join("preprocessed", "metadata.yaml")
    if os.path.exists(os.path.join(dataset_dir, preprocess_metadata_path)):
        print("The dataset is already preprocessed.")
        return os.path.join(dataset_dir, preprocess_metadata_path)

    print("Start to preprocess the on-disk dataset.")
    processed_dir_prefix = "preprocessed"

    # Check if the metadata.yaml exists.
    metadata_file_path = os.path.join(dataset_dir, "metadata.yaml")
    if not os.path.exists(metadata_file_path):
        raise RuntimeError("metadata.yaml does not exist.")

    # Read the input config.
    with open(metadata_file_path, "r") as f:
        input_config = yaml.safe_load(f)

    # 1. Make `processed_dir_abs` directory if it does not exist.
    os.makedirs(os.path.join(dataset_dir, processed_dir_prefix), exist_ok=True)
    output_config = deepcopy(input_config)

    # 2. Load the edge data and create a DGLGraph.
    if "graph" not in input_config:
        raise RuntimeError("Invalid config: does not contain graph field.")
    is_homogeneous = "type" not in input_config["graph"]["nodes"][0]
    if is_homogeneous:
        # Homogeneous graph.
        num_nodes = input_config["graph"]["nodes"][0]["num"]
        edge_data = pd.read_csv(
            os.path.join(
                dataset_dir, input_config["graph"]["edges"][0]["path"]
            ),
            names=["src", "dst"],
        )
        src, dst = edge_data["src"].to_numpy(), edge_data["dst"].to_numpy()

        g = dgl.graph((src, dst), num_nodes=num_nodes)
    else:
        # Heterogeneous graph.
        # Construct the num nodes dict.
        num_nodes_dict = {}
        for node_info in input_config["graph"]["nodes"]:
            num_nodes_dict[node_info["type"]] = node_info["num"]
        # Construct the data dict.
        data_dict = {}
        for edge_info in input_config["graph"]["edges"]:
            edge_data = pd.read_csv(
                os.path.join(dataset_dir, edge_info["path"]),
                names=["src", "dst"],
            )
            src = torch.tensor(edge_data["src"])
            dst = torch.tensor(edge_data["dst"])
            data_dict[tuple(edge_info["type"].split(":"))] = (src, dst)
        # Construct the heterograph.
        g = dgl.heterograph(data_dict, num_nodes_dict)

    # 3. Load the sampling related node/edge features and add them to
    # the sampling-graph.
    if input_config["graph"].get("feature_data", None):
        for graph_feature in input_config["graph"]["feature_data"]:
            if graph_feature["domain"] == "node":
                node_data = read_data(
                    os.path.join(dataset_dir, graph_feature["path"]),
                    graph_feature["format"],
                    in_memory=graph_feature["in_memory"],
                )
                g.ndata[graph_feature["name"]] = node_data
            if graph_feature["domain"] == "edge":
                edge_data = read_data(
                    os.path.join(dataset_dir, graph_feature["path"]),
                    graph_feature["format"],
                    in_memory=graph_feature["in_memory"],
                )
                g.edata[graph_feature["name"]] = edge_data

    # 4. Convert the DGLGraph to a CSCSamplingGraph.
    csc_sampling_graph = from_dglgraph(g)

    # 5. Save the CSCSamplingGraph and modify the output_config.
    output_config["graph_topology"] = {}
    output_config["graph_topology"]["type"] = "CSCSamplingGraph"
    output_config["graph_topology"]["path"] = os.path.join(
        processed_dir_prefix, "csc_sampling_graph.tar"
    )

    save_csc_sampling_graph(
        csc_sampling_graph,
        os.path.join(
            dataset_dir,
            output_config["graph_topology"]["path"],
        ),
    )
    del output_config["graph"]

    # 6. Load the node/edge features and do necessary conversion.
    if input_config.get("feature_data", None):
        for feature, out_feature in zip(
            input_config["feature_data"], output_config["feature_data"]
        ):
            # Always save the feature in numpy format.
            out_feature["format"] = "numpy"
            out_feature["path"] = os.path.join(
                processed_dir_prefix, feature["path"].replace("pt", "npy")
            )
            _copy_or_convert_data(
                os.path.join(dataset_dir, feature["path"]),
                os.path.join(dataset_dir, out_feature["path"]),
                feature["format"],
                out_feature["format"],
                feature["in_memory"],
            )

    # 7. Save tasks and train/val/test split according to the output_config.
    if input_config.get("tasks", None):
        for input_task, output_task in zip(
            input_config["tasks"], output_config["tasks"]
        ):
            for set_name in ["train_set", "validation_set", "test_set"]:
                if set_name not in input_task:
                    continue
                for input_set_per_type, output_set_per_type in zip(
                    input_task[set_name], output_task[set_name]
                ):
                    for input_data, output_data in zip(
                        input_set_per_type["data"], output_set_per_type["data"]
                    ):
                        # Always save the feature in numpy format.
                        output_data["format"] = "numpy"
                        output_data["path"] = os.path.join(
                            processed_dir_prefix,
                            input_data["path"].replace("pt", "npy"),
                        )
                        _copy_or_convert_data(
                            os.path.join(dataset_dir, input_data["path"]),
                            os.path.join(dataset_dir, output_data["path"]),
                            input_data["format"],
                            output_data["format"],
                        )

    # 8. Save the output_config.
    output_config_path = os.path.join(dataset_dir, preprocess_metadata_path)
    with open(output_config_path, "w") as f:
        yaml.dump(output_config, f)
    print("Finish preprocessing the on-disk dataset.")

    # 9. Return the absolute path of the preprocessing yaml file.
    return output_config_path


class OnDiskTask:
    """An on-disk task.

    An on-disk task is for ``OnDiskDataset``. It contains the metadata and the
    train/val/test sets.
    """

    def __init__(
        self,
        metadata: Dict,
        train_set: ItemSet or ItemSetDict,
        validation_set: ItemSet or ItemSetDict,
        test_set: ItemSet or ItemSetDict,
    ):
        """Initialize a task.

        Parameters
        ----------
        metadata : Dict
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
    def metadata(self) -> Dict:
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


class OnDiskDataset(Dataset):
    """An on-disk dataset.

    An on-disk dataset is a dataset which reads graph topology, feature data
    and TVT set from disk. Due to limited resources, the data which are too
    large to fit into RAM will remain on disk while others reside in RAM once
    ``OnDiskDataset`` is initialized. This behavior could be controled by user
    via ``in_memory`` field in YAML file.

    A full example of YAML file is as follows:

    .. code-block:: yaml

        dataset_name: graphbolt_test
        graph_topology:
          type: CSCSamplingGraph
          path: graph_topology/csc_sampling_graph.tar
        feature_data:
          - domain: node
            type: paper
            name: feat
            format: numpy
            in_memory: false
            path: node_data/paper-feat.npy
          - domain: edge
            type: "author:writes:paper"
            name: feat
            format: numpy
            in_memory: false
            path: edge_data/author-writes-paper-feat.npy
        tasks:
          - name: "edge_classification"
            num_classes: 10
            train_set:
              - type: paper # could be null for homogeneous graph.
                data: # multiple data sources could be specified.
                  - format: numpy
                    in_memory: true # If not specified, default to true.
                    path: set/paper-train-src.npy
                  - format: numpy
                    in_memory: false
                    path: set/paper-train-dst.npy
            validation_set:
              - type: paper
                data:
                  - format: numpy
                    in_memory: true
                    path: set/paper-validation.npy
            test_set:
              - type: paper
                data:
                  - format: numpy
                    in_memory: true
                    path: set/paper-test.npy

    Parameters
    ----------
    path: str
        The YAML file path.
    """

    def __init__(self, path: str) -> None:
        # Always call the preprocess function first. If already preprocessed,
        # the function will return the original path directly.
        self.dataset_dir = path
        path = preprocess_ondisk_dataset(path)
        with open(path) as f:
            self.yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)
        self._convert_yaml_path_to_absolute_path()
        self._meta = OnDiskMetaData(**self.yaml_data)
        self._dataset_name = self._meta.dataset_name
        self._graph = self._load_graph(self._meta.graph_topology)
        self._feature = TorchBasedFeatureStore(self._meta.feature_data)
        self._tasks = self._init_tasks(self._meta.tasks)

    def _convert_yaml_path_to_absolute_path(self):
        """Convert the path in YAML file to absolute path."""
        if "graph_topology" in self.yaml_data:
            self.yaml_data["graph_topology"]["path"] = os.path.join(
                self.dataset_dir, self.yaml_data["graph_topology"]["path"]
            )
        if "feature_data" in self.yaml_data:
            for feature in self.yaml_data["feature_data"]:
                feature["path"] = os.path.join(
                    self.dataset_dir, feature["path"]
                )
        if "tasks" in self.yaml_data:
            for task in self.yaml_data["tasks"]:
                for set_name in ["train_set", "validation_set", "test_set"]:
                    if set_name not in task:
                        continue
                    for set_per_type in task[set_name]:
                        for data in set_per_type["data"]:
                            data["path"] = os.path.join(
                                self.dataset_dir, data["path"]
                            )

    @property
    def tasks(self) -> List[Task]:
        """Return the tasks."""
        return self._tasks

    @property
    def graph(self) -> object:
        """Return the graph."""
        return self._graph

    @property
    def feature(self) -> TorchBasedFeatureStore:
        """Return the feature."""
        return self._feature

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return self._dataset_name

    def _init_tasks(self, tasks: List[OnDiskTaskData]) -> List[OnDiskTask]:
        """Initialize the tasks."""
        ret = []
        if tasks is None:
            return ret
        for task in tasks:
            ret.append(
                OnDiskTask(
                    task.extra_fields,
                    self._init_tvt_set(task.train_set),
                    self._init_tvt_set(task.validation_set),
                    self._init_tvt_set(task.test_set),
                )
            )
        return ret

    def _load_graph(
        self, graph_topology: OnDiskGraphTopology
    ) -> CSCSamplingGraph:
        """Load the graph topology."""
        if graph_topology is None:
            return None
        if graph_topology.type == "CSCSamplingGraph":
            return load_csc_sampling_graph(graph_topology.path)
        raise NotImplementedError(
            f"Graph topology type {graph_topology.type} is not supported."
        )

    def _init_tvt_set(
        self, tvt_set: List[OnDiskTVTSet]
    ) -> ItemSet or ItemSetDict:
        """Initialize the TVT set."""
        ret = None
        if (tvt_set is None) or (len(tvt_set) == 0):
            return ret
        if tvt_set[0].type is None:
            assert (
                len(tvt_set) == 1
            ), "Only one TVT set is allowed if type is not specified."
            ret = ItemSet(
                tuple(
                    read_data(data.path, data.format, data.in_memory)
                    for data in tvt_set[0].data
                )
            )
        else:
            data = {}
            for tvt in tvt_set:
                data[tvt.type] = ItemSet(
                    tuple(
                        read_data(data.path, data.format, data.in_memory)
                        for data in tvt.data
                    )
                )
            ret = ItemSetDict(data)
        return ret
