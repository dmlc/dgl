"""GraphBolt OnDiskDataset."""

import os
import shutil

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import yaml

import dgl

from ..dataset import Dataset
from ..itemset import ItemSet, ItemSetDict
from ..utils import read_data, save_data

from .csc_sampling_graph import (
    CSCSamplingGraph,
    from_dglgraph,
    load_csc_sampling_graph,
    save_csc_sampling_graph,
)
from .ondisk_metadata import OnDiskGraphTopology, OnDiskMetaData, OnDiskTVTSet
from .torch_based_feature_store import (
    load_feature_stores,
    TorchBasedFeatureStore,
)

__all__ = ["OnDiskDataset", "preprocess_ondisk_dataset"]


def preprocess_ondisk_dataset(input_config_path: str) -> str:
    """Preprocess the on-disk dataset. Parse the input config file,
    load the data, and save the data in the format that GraphBolt supports.

    Parameters
    ----------
    input_config_path : str
        The path to the input config file.

    Returns
    -------
    output_config_path : str
        The path to the output config file.
    """
    # 0. Load the input_config.
    with open(input_config_path, "r") as f:
        input_config = yaml.safe_load(f)

    # If the input config does not contain the "graph" field, then we
    # assume that the input config is already preprocessed.
    if "graph" not in input_config:
        print("The input config is already preprocessed.")
        return input_config_path

    print("Start to preprocess the on-disk dataset.")
    # Infer the dataset path from the input config path.
    dataset_path = Path(os.path.dirname(input_config_path))
    processed_dir_prefix = Path("preprocessed")

    # 1. Make `processed_dir_prefix` directory if it does not exist.
    os.makedirs(dataset_path / processed_dir_prefix, exist_ok=True)
    output_config = deepcopy(input_config)

    # 2. Load the edge data and create a DGLGraph.
    is_homogeneous = "type" not in input_config["graph"]["nodes"][0]
    if is_homogeneous:
        # Homogeneous graph.
        num_nodes = input_config["graph"]["nodes"][0]["num"]
        edge_data = pd.read_csv(
            dataset_path / input_config["graph"]["edges"][0]["path"],
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
                dataset_path / edge_info["path"], names=["src", "dst"]
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
                    dataset_path / graph_feature["path"],
                    graph_feature["format"],
                    in_memory=graph_feature["in_memory"],
                )
                g.ndata[graph_feature["name"]] = node_data
            if graph_feature["domain"] == "edge":
                edge_data = read_data(
                    dataset_path / graph_feature["path"],
                    graph_feature["format"],
                    in_memory=graph_feature["in_memory"],
                )
                g.edata[graph_feature["name"]] = edge_data

    # 4. Convert the DGLGraph to a CSCSamplingGraph.
    csc_sampling_graph = from_dglgraph(g)

    # 5. Save the CSCSamplingGraph and modify the output_config.
    output_config["graph_topology"] = {}
    output_config["graph_topology"]["type"] = "CSCSamplingGraph"
    output_config["graph_topology"]["path"] = str(
        processed_dir_prefix / "csc_sampling_graph.tar"
    )

    save_csc_sampling_graph(
        csc_sampling_graph,
        str(dataset_path / output_config["graph_topology"]["path"]),
    )
    del output_config["graph"]

    # 6. Load the node/edge features and do necessary conversion.
    if input_config.get("feature_data", None):
        for feature, out_feature in zip(
            input_config["feature_data"], output_config["feature_data"]
        ):
            # Always save the feature in numpy format.
            out_feature["format"] = "numpy"
            out_feature["path"] = str(
                processed_dir_prefix / feature["path"].replace("pt", "npy")
            )

            if feature["format"] == "numpy":
                # If the original format is numpy, just copy the file.
                os.makedirs(
                    dataset_path / os.path.dirname(out_feature["path"]),
                    exist_ok=True,
                )
                shutil.copyfile(
                    dataset_path / feature["path"],
                    dataset_path / out_feature["path"],
                )
            else:
                # If the original format is not numpy, convert it to numpy.
                data = read_data(
                    dataset_path / feature["path"],
                    feature["format"],
                    in_memory=feature["in_memory"],
                )
                save_data(
                    data,
                    dataset_path / out_feature["path"],
                    out_feature["format"],
                )

    # 7. Save the train/val/test split according to the output_config.
    for set_name in ["train_sets", "validation_sets", "test_sets"]:
        if set_name not in input_config:
            continue
        for intput_set_split, output_set_split in zip(
            input_config[set_name], output_config[set_name]
        ):
            for input_set_per_type, output_set_per_type in zip(
                intput_set_split, output_set_split
            ):
                for input_data, output_data in zip(
                    input_set_per_type["data"], output_set_per_type["data"]
                ):
                    # Always save the feature in numpy format.
                    output_data["format"] = "numpy"
                    output_data["path"] = str(
                        processed_dir_prefix
                        / input_data["path"].replace("pt", "npy")
                    )
                    if input_data["format"] == "numpy":
                        # If the original format is numpy, just copy the file.
                        os.makedirs(
                            dataset_path / os.path.dirname(output_data["path"]),
                            exist_ok=True,
                        )
                        shutil.copy(
                            dataset_path / input_data["path"],
                            dataset_path / output_data["path"],
                        )
                    else:
                        # If the original format is not numpy, convert it to numpy.
                        input_set = read_data(
                            dataset_path / input_data["path"],
                            input_data["format"],
                        )
                        save_data(
                            input_set,
                            dataset_path / output_data["path"],
                            output_set_per_type["format"],
                        )

    # 8. Save the output_config.
    output_config_path = dataset_path / "output_config.yaml"
    with open(output_config_path, "w") as f:
        yaml.dump(output_config, f)
    print("Finish preprocessing the on-disk dataset.")
    return str(output_config_path)


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
        num_classes: 10
        num_labels: 10
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
        train_sets:
          - - type: paper # could be null for homogeneous graph.
              data: # multiple data sources could be specified.
                - format: numpy
                  in_memory: true # If not specified, default to true.
                  path: set/paper-train-src.npy
                - format: numpy
                  in_memory: false
                  path: set/paper-train-dst.npy
        validation_sets:
          - - type: paper
              data:
                - format: numpy
                  in_memory: true
                  path: set/paper-validation.npy
        test_sets:
          - - type: paper
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
        path = preprocess_ondisk_dataset(path)
        with open(path) as f:
            yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)
            self._meta = OnDiskMetaData(**yaml_data)
        self._dataset_name = self._meta.dataset_name
        self._num_classes = self._meta.num_classes
        self._num_labels = self._meta.num_labels
        self._graph = self._load_graph(self._meta.graph_topology)
        self._feature = load_feature_stores(self._meta.feature_data)
        self._train_sets = self._init_tvt_sets(self._meta.train_sets)
        self._validation_sets = self._init_tvt_sets(self._meta.validation_sets)
        self._test_sets = self._init_tvt_sets(self._meta.test_sets)

    @property
    def train_sets(self) -> List[ItemSet] or List[ItemSetDict]:
        """Return the training set."""
        return self._train_sets

    @property
    def validation_sets(self) -> List[ItemSet] or List[ItemSetDict]:
        """Return the validation set."""
        return self._validation_sets

    @property
    def test_sets(self) -> List[ItemSet] or List[ItemSetDict]:
        """Return the test set."""
        return self._test_sets

    @property
    def graph(self) -> object:
        """Return the graph."""
        return self._graph

    @property
    def feature(self) -> Dict[Tuple, TorchBasedFeatureStore]:
        """Return the feature."""
        return self._feature

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return self._dataset_name

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return self._num_classes

    @property
    def num_labels(self) -> int:
        """Return the number of labels."""
        return self._num_labels

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
            if tvt_set[0].type is None:
                assert (
                    len(tvt_set) == 1
                ), "Only one TVT set is allowed if type is not specified."
                ret.append(
                    ItemSet(
                        tuple(
                            read_data(data.path, data.format, data.in_memory)
                            for data in tvt_set[0].data
                        )
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
                ret.append(ItemSetDict(data))
        return ret
