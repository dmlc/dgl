"""GraphBolt OnDiskDataset."""

import os
from copy import deepcopy
from typing import Dict, List, Union

import pandas as pd
import torch
import yaml

import dgl

from ...base import dgl_warning
from ...data.utils import download, extract_archive
from ..base import etype_str_to_tuple
from ..dataset import Dataset, Task
from ..internal import copy_or_convert_data, get_attributes, read_data
from ..itemset import ItemSet, ItemSetDict
from ..sampling_graph import SamplingGraph
from .fused_csc_sampling_graph import from_dglgraph, FusedCSCSamplingGraph
from .ondisk_metadata import (
    OnDiskGraphTopology,
    OnDiskMetaData,
    OnDiskTaskData,
    OnDiskTVTSet,
)
from .torch_based_feature_store import TorchBasedFeatureStore

__all__ = ["OnDiskDataset", "preprocess_ondisk_dataset", "BuiltinDataset"]


def preprocess_ondisk_dataset(
    dataset_dir: str, include_original_edge_id: bool = False
) -> str:
    """Preprocess the on-disk dataset. Parse the input config file,
    load the data, and save the data in the format that GraphBolt supports.

    Parameters
    ----------
    dataset_dir : str
        The path to the dataset directory.
    include_original_edge_id : bool, optional
        Whether to include the original edge id in the FusedCSCSamplingGraph.

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
            data_dict[etype_str_to_tuple(edge_info["type"])] = (src, dst)
        # Construct the heterograph.
        g = dgl.heterograph(data_dict, num_nodes_dict)

    # 3. Load the sampling related node/edge features and add them to
    # the sampling-graph.
    if input_config["graph"].get("feature_data", None):
        for graph_feature in input_config["graph"]["feature_data"]:
            in_memory = (
                True
                if "in_memory" not in graph_feature
                else graph_feature["in_memory"]
            )
            if graph_feature["domain"] == "node":
                node_data = read_data(
                    os.path.join(dataset_dir, graph_feature["path"]),
                    graph_feature["format"],
                    in_memory=in_memory,
                )
                g.ndata[graph_feature["name"]] = node_data
            if graph_feature["domain"] == "edge":
                edge_data = read_data(
                    os.path.join(dataset_dir, graph_feature["path"]),
                    graph_feature["format"],
                    in_memory=in_memory,
                )
                g.edata[graph_feature["name"]] = edge_data

    # 4. Convert the DGLGraph to a FusedCSCSamplingGraph.
    fused_csc_sampling_graph = from_dglgraph(
        g, is_homogeneous, include_original_edge_id
    )

    # 5. Save the FusedCSCSamplingGraph and modify the output_config.
    output_config["graph_topology"] = {}
    output_config["graph_topology"]["type"] = "FusedCSCSamplingGraph"
    output_config["graph_topology"]["path"] = os.path.join(
        processed_dir_prefix, "fused_csc_sampling_graph.pt"
    )

    torch.save(
        fused_csc_sampling_graph,
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
            in_memory = (
                True if "in_memory" not in feature else feature["in_memory"]
            )
            copy_or_convert_data(
                os.path.join(dataset_dir, feature["path"]),
                os.path.join(dataset_dir, out_feature["path"]),
                feature["format"],
                output_format=out_feature["format"],
                in_memory=in_memory,
                is_feature=True,
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
                        copy_or_convert_data(
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
        train_set: Union[ItemSet, ItemSetDict],
        validation_set: Union[ItemSet, ItemSetDict],
        test_set: Union[ItemSet, ItemSetDict],
    ):
        """Initialize a task.

        Parameters
        ----------
        metadata : Dict
            Metadata.
        train_set : Union[ItemSet, ItemSetDict]
            Training set.
        validation_set : Union[ItemSet, ItemSetDict]
            Validation set.
        test_set : Union[ItemSet, ItemSetDict]
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

    def __repr__(self) -> str:
        return _ondisk_task_str(self)


class OnDiskDataset(Dataset):
    """An on-disk dataset which reads graph topology, feature data and
    Train/Validation/Test set from disk.

    Due to limited resources, the data which are too large to fit into RAM will
    remain on disk while others reside in RAM once ``OnDiskDataset`` is
    initialized. This behavior could be controled by user via ``in_memory``
    field in YAML file. All paths in YAML file are relative paths to the
    dataset directory.

    A full example of YAML file is as follows:

    .. code-block:: yaml

        dataset_name: graphbolt_test
        graph:
          nodes:
            - type: paper # could be omitted for homogeneous graph.
              num: 1000
            - type: author
              num: 1000
          edges:
            - type: author:writes:paper # could be omitted for homogeneous graph.
              format: csv # Can be csv only.
              path: edge_data/author-writes-paper.csv
            - type: paper:cites:paper
              format: csv
              path: edge_data/paper-cites-paper.csv
        feature_data:
          - domain: node
            type: paper # could be omitted for homogeneous graph.
            name: feat
            format: numpy
            in_memory: false # If not specified, default to true.
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
              - type: paper # could be omitted for homogeneous graph.
                data: # multiple data sources could be specified.
                  - name: node_pairs
                    format: numpy # Can be numpy or torch.
                    in_memory: true # If not specified, default to true.
                    path: set/paper-train-node_pairs.npy
                  - name: labels
                    format: numpy
                    path: set/paper-train-labels.npy
            validation_set:
              - type: paper
                data:
                  - name: node_pairs
                    format: numpy
                    path: set/paper-validation-node_pairs.npy
                  - name: labels
                    format: numpy
                    path: set/paper-validation-labels.npy
            test_set:
              - type: paper
                data:
                  - name: node_pairs
                    format: numpy
                    path: set/paper-test-node_pairs.npy
                  - name: labels
                    format: numpy
                    path: set/paper-test-labels.npy

    Parameters
    ----------
    path: str
        The YAML file path.
    include_original_edge_id: bool, optional
        Whether to include the original edge id in the FusedCSCSamplingGraph.
    """

    def __init__(
        self, path: str, include_original_edge_id: bool = False
    ) -> None:
        # Always call the preprocess function first. If already preprocessed,
        # the function will return the original path directly.
        self._dataset_dir = path
        yaml_path = preprocess_ondisk_dataset(path, include_original_edge_id)
        with open(yaml_path) as f:
            self._yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)
        self._loaded = False

    def _convert_yaml_path_to_absolute_path(self):
        """Convert the path in YAML file to absolute path."""
        if "graph_topology" in self._yaml_data:
            self._yaml_data["graph_topology"]["path"] = os.path.join(
                self._dataset_dir, self._yaml_data["graph_topology"]["path"]
            )
        if "feature_data" in self._yaml_data:
            for feature in self._yaml_data["feature_data"]:
                feature["path"] = os.path.join(
                    self._dataset_dir, feature["path"]
                )
        if "tasks" in self._yaml_data:
            for task in self._yaml_data["tasks"]:
                for set_name in ["train_set", "validation_set", "test_set"]:
                    if set_name not in task:
                        continue
                    for set_per_type in task[set_name]:
                        for data in set_per_type["data"]:
                            data["path"] = os.path.join(
                                self._dataset_dir, data["path"]
                            )

    def load(self):
        """Load the dataset."""
        self._convert_yaml_path_to_absolute_path()
        self._meta = OnDiskMetaData(**self._yaml_data)
        self._dataset_name = self._meta.dataset_name
        self._graph = self._load_graph(self._meta.graph_topology)
        self._feature = TorchBasedFeatureStore(self._meta.feature_data)
        self._tasks = self._init_tasks(self._meta.tasks)
        self._all_nodes_set = self._init_all_nodes_set(self._graph)
        self._loaded = True
        return self

    @property
    def yaml_data(self) -> Dict:
        """Return the YAML data."""
        return self._yaml_data

    @property
    def tasks(self) -> List[Task]:
        """Return the tasks."""
        self._check_loaded()
        return self._tasks

    @property
    def graph(self) -> SamplingGraph:
        """Return the graph."""
        self._check_loaded()
        return self._graph

    @property
    def feature(self) -> TorchBasedFeatureStore:
        """Return the feature."""
        self._check_loaded()
        return self._feature

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        self._check_loaded()
        return self._dataset_name

    @property
    def all_nodes_set(self) -> Union[ItemSet, ItemSetDict]:
        """Return the itemset containing all nodes."""
        self._check_loaded()
        return self._all_nodes_set

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

    def _check_loaded(self):
        assert self._loaded, (
            "Please ensure that you have called the OnDiskDataset.load() method"
            + " to properly load the data."
        )

    def _load_graph(
        self, graph_topology: OnDiskGraphTopology
    ) -> FusedCSCSamplingGraph:
        """Load the graph topology."""
        if graph_topology is None:
            return None
        if graph_topology.type == "FusedCSCSamplingGraph":
            return torch.load(graph_topology.path)
        raise NotImplementedError(
            f"Graph topology type {graph_topology.type} is not supported."
        )

    def _init_tvt_set(
        self, tvt_set: List[OnDiskTVTSet]
    ) -> Union[ItemSet, ItemSetDict]:
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
                ),
                names=tuple(data.name for data in tvt_set[0].data),
            )
        else:
            data = {}
            for tvt in tvt_set:
                data[tvt.type] = ItemSet(
                    tuple(
                        read_data(data.path, data.format, data.in_memory)
                        for data in tvt.data
                    ),
                    names=tuple(data.name for data in tvt.data),
                )
            ret = ItemSetDict(data)
        return ret

    def _init_all_nodes_set(self, graph) -> Union[ItemSet, ItemSetDict]:
        if graph is None:
            dgl_warning(
                "`all_node_set` is returned as None, since graph is None."
            )
            return None
        num_nodes = graph.num_nodes
        if isinstance(num_nodes, int):
            return ItemSet(num_nodes, names="seed_nodes")
        else:
            data = {
                node_type: ItemSet(num_node, names="seed_nodes")
                for node_type, num_node in num_nodes.items()
            }
            return ItemSetDict(data)


class BuiltinDataset(OnDiskDataset):
    """A utility class to download built-in dataset from AWS S3 and load it as
    :class:`OnDiskDataset`.

    Available built-in datasets include:

    **cora**
        The cora dataset is a homogeneous citation network dataset, which is
        designed for the node classification task.

    **ogbn-mag**
        The ogbn-mag dataset is a heterogeneous network composed of a subset of
        the Microsoft Academic Graph (MAG). See more details in
        `ogbn-mag <https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag>`_.

        .. note::
            Reverse edges are added to the original graph and duplicated
            edges are removed.

    **ogbl-citation2**
        The ogbl-citation2 dataset is a directed graph, representing the
        citation network between a subset of papers extracted from MAG. See
        more details in `ogbl-citation2
        <https://ogb.stanford.edu/docs/linkprop/#ogbl-citation2>`_.

        .. note::
            Reverse edges are added to the original graph and duplicated
            edges are removed.

    **ogbn-arxiv**
        The ogbn-arxiv dataset is a directed graph, representing the citation
        network between all Computer Science (CS) arXiv papers indexed by MAG.
        See more details in `ogbn-arxiv
        <https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv>`_.

        .. note::
            Reverse edges are added to the original graph and duplicated
            edges are removed.

    **ogbn-products**
        The ogbn-products dataset is an undirected and unweighted graph,
        representing an Amazon product co-purchasing network. See more details
        in `ogbn-products
        <https://ogb.stanford.edu/docs/nodeprop/#ogbn-products>`_.

        .. note::
            Reverse edges are added to the original graph.
            Node features are stored as float32.

    **ogb-lsc-mag240m**
        The ogb-lsc-mag240m dataset is a heterogeneous academic graph extracted
        from the Microsoft Academic Graph (MAG). See more details in
        `ogb-lsc-mag240m <https://ogb.stanford.edu/docs/lsc/mag240m/>`_.

        .. note::
            Reverse edges are added to the original graph.

    Parameters
    ----------
    name : str
        The name of the builtin dataset.
    root : str, optional
        The root directory of the dataset. Default ot ``datasets``.
    """

    # For dataset that is smaller than 30GB, we use the base url.
    # Otherwise, we use the accelerated url.
    _base_url = "https://data.dgl.ai/dataset/graphbolt/"
    _accelerated_url = (
        "https://dgl-data.s3-accelerate.amazonaws.com/dataset/graphbolt/"
    )
    _datasets = [
        "cora",
        "ogbn-mag",
        "ogbl-citation2",
        "ogbn-products",
        "ogbn-arxiv",
    ]
    _large_datasets = ["ogb-lsc-mag240m"]
    _all_datasets = _datasets + _large_datasets

    def __init__(self, name: str, root: str = "datasets") -> OnDiskDataset:
        dataset_dir = os.path.join(root, name)
        if not os.path.exists(dataset_dir):
            if name not in self._all_datasets:
                raise RuntimeError(
                    f"Dataset {name} is not available. Available datasets are "
                    f"{self._all_datasets}."
                )
            url = (
                self._accelerated_url
                if name in self._large_datasets
                else self._base_url
            )
            url += name + ".zip"
            os.makedirs(root, exist_ok=True)
            zip_file_path = os.path.join(root, name + ".zip")
            download(url, path=zip_file_path)
            extract_archive(zip_file_path, root, overwrite=True)
            os.remove(zip_file_path)
        super().__init__(dataset_dir)


def _ondisk_task_str(task: OnDiskTask) -> str:
    final_str = "OnDiskTask("
    indent_len = len(final_str)

    def _add_indent(_str, indent):
        lines = _str.split("\n")
        lines = [lines[0]] + [" " * indent + line for line in lines[1:]]
        return "\n".join(lines)

    attributes = get_attributes(task)
    attributes.reverse()
    for name in attributes:
        if name[0] == "_":
            continue
        val = getattr(task, name)
        final_str += (
            f"{name}={_add_indent(str(val), indent_len + len(name) + 1)},\n"
            + " " * indent_len
        )
    return final_str[:-indent_len] + ")"
