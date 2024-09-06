"""GraphBolt OnDiskDataset."""

import bisect
import json
import os
import shutil
import textwrap
from copy import deepcopy
from typing import Dict, List, Union

import numpy as np

import torch
import yaml

from ..base import etype_str_to_tuple, ORIGINAL_EDGE_ID
from ..dataset import Dataset, Task
from ..internal import (
    calculate_dir_hash,
    check_dataset_change,
    copy_or_convert_data,
    read_data,
    read_edges,
)
from ..internal_utils import (
    download,
    extract_archive,
    gb_warning,
    get_attributes,
)
from ..itemset import HeteroItemSet, ItemSet
from ..sampling_graph import SamplingGraph
from .fused_csc_sampling_graph import (
    fused_csc_sampling_graph,
    FusedCSCSamplingGraph,
)
from .ondisk_metadata import (
    OnDiskGraphTopology,
    OnDiskMetaData,
    OnDiskTaskData,
    OnDiskTVTSet,
)
from .torch_based_feature_store import TorchBasedFeatureStore

__all__ = ["OnDiskDataset", "preprocess_ondisk_dataset", "BuiltinDataset"]

NAMES_INDICATING_NODE_IDS = [
    "seeds",
]


def _graph_data_to_fused_csc_sampling_graph(
    dataset_dir: str,
    graph_data: Dict,
    include_original_edge_id: bool,
    auto_cast_to_optimal_dtype: bool,
) -> FusedCSCSamplingGraph:
    """Convert the raw graph data into FusedCSCSamplingGraph.

    Parameters
    ----------
    dataset_dir : str
        The path to the dataset directory.
    graph_data : Dict
        The raw data read from yaml file.
    include_original_edge_id : bool
        Whether to include the original edge id in the FusedCSCSamplingGraph.
    auto_cast_to_optimal_dtype: bool, optional
        Casts the dtypes of tensors in the dataset into smallest possible dtypes
        for reduced storage requirements and potentially increased performance.

    Returns
    -------
    sampling_graph : FusedCSCSamplingGraph
        The FusedCSCSamplingGraph constructed from the raw data.
    """
    from ...sparse import spmatrix

    is_homogeneous = (
        len(graph_data["nodes"]) == 1
        and len(graph_data["edges"]) == 1
        and "type" not in graph_data["nodes"][0]
        and "type" not in graph_data["edges"][0]
    )

    if is_homogeneous:
        # Homogeneous graph.
        edge_fmt = graph_data["edges"][0]["format"]
        edge_path = graph_data["edges"][0]["path"]
        src, dst = read_edges(dataset_dir, edge_fmt, edge_path)
        num_nodes = graph_data["nodes"][0]["num"]
        num_edges = len(src)
        coo_tensor = torch.tensor(np.array([src, dst]))
        sparse_matrix = spmatrix(coo_tensor, shape=(num_nodes, num_nodes))
        del coo_tensor
        indptr, indices, edge_ids = sparse_matrix.csc()
        del sparse_matrix

        if auto_cast_to_optimal_dtype:
            if num_nodes <= torch.iinfo(torch.int32).max:
                indices = indices.to(torch.int32)
            if num_edges <= torch.iinfo(torch.int32).max:
                indptr = indptr.to(torch.int32)
                edge_ids = edge_ids.to(torch.int32)

        node_type_offset = None
        type_per_edge = None
        node_type_to_id = None
        edge_type_to_id = None
        node_attributes = {}
        edge_attributes = {}
        if include_original_edge_id:
            edge_attributes[ORIGINAL_EDGE_ID] = edge_ids
    else:
        # Heterogeneous graph.
        # Sort graph_data by ntype/etype lexicographically to ensure ordering.
        graph_data["nodes"].sort(key=lambda x: x["type"])
        graph_data["edges"].sort(key=lambda x: x["type"])
        # Construct node_type_offset and node_type_to_id.
        node_type_offset = [0]
        node_type_to_id = {}
        for ntype_id, node_info in enumerate(graph_data["nodes"]):
            node_type_to_id[node_info["type"]] = ntype_id
            node_type_offset.append(node_type_offset[-1] + node_info["num"])
        total_num_nodes = node_type_offset[-1]
        # Construct edge_type_offset, edge_type_to_id and coo_tensor.
        edge_type_offset = [0]
        edge_type_to_id = {}
        coo_src_list = []
        coo_dst_list = []
        coo_etype_list = []
        for etype_id, edge_info in enumerate(graph_data["edges"]):
            edge_type_to_id[edge_info["type"]] = etype_id
            edge_fmt = edge_info["format"]
            edge_path = edge_info["path"]
            src, dst = read_edges(dataset_dir, edge_fmt, edge_path)
            edge_type_offset.append(edge_type_offset[-1] + len(src))
            src_type, _, dst_type = etype_str_to_tuple(edge_info["type"])
            src += node_type_offset[node_type_to_id[src_type]]
            dst += node_type_offset[node_type_to_id[dst_type]]
            coo_src_list.append(torch.tensor(src))
            coo_dst_list.append(torch.tensor(dst))
            coo_etype_list.append(torch.full((len(src),), etype_id))
        total_num_edges = edge_type_offset[-1]

        coo_src = torch.cat(coo_src_list)
        del coo_src_list
        coo_dst = torch.cat(coo_dst_list)
        del coo_dst_list
        if auto_cast_to_optimal_dtype:
            dtypes = [torch.uint8, torch.int16, torch.int32, torch.int64]
            dtype_maxes = [torch.iinfo(dtype).max for dtype in dtypes]
            dtype_id = bisect.bisect_left(dtype_maxes, len(edge_type_to_id) - 1)
            etype_dtype = dtypes[dtype_id]
            coo_etype_list = [
                tensor.to(etype_dtype) for tensor in coo_etype_list
            ]
        coo_etype = torch.cat(coo_etype_list)
        del coo_etype_list

        sparse_matrix = spmatrix(
            indices=torch.stack((coo_src, coo_dst), dim=0),
            shape=(total_num_nodes, total_num_nodes),
        )
        del coo_src, coo_dst
        indptr, indices, edge_ids = sparse_matrix.csc()
        del sparse_matrix

        if auto_cast_to_optimal_dtype:
            if total_num_nodes <= torch.iinfo(torch.int32).max:
                indices = indices.to(torch.int32)
            if total_num_edges <= torch.iinfo(torch.int32).max:
                indptr = indptr.to(torch.int32)
                edge_ids = edge_ids.to(torch.int32)

        node_type_offset = torch.tensor(node_type_offset, dtype=indices.dtype)
        type_per_edge = torch.index_select(coo_etype, dim=0, index=edge_ids)
        del coo_etype
        node_attributes = {}
        edge_attributes = {}
        if include_original_edge_id:
            # If uint8 or int16 was chosen above for etypes, we cast to int.
            temp_etypes = (
                type_per_edge.int()
                if type_per_edge.element_size() < 4
                else type_per_edge
            )
            edge_ids -= torch.index_select(
                torch.tensor(edge_type_offset, dtype=edge_ids.dtype),
                dim=0,
                index=temp_etypes,
            )
            del temp_etypes
            edge_attributes[ORIGINAL_EDGE_ID] = edge_ids

    # Load the sampling related node/edge features and add them to
    # the sampling-graph.
    if graph_data.get("feature_data", None):
        if is_homogeneous:
            # Homogeneous graph.
            for graph_feature in graph_data["feature_data"]:
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
                    assert node_data.shape[0] == num_nodes
                    node_attributes[graph_feature["name"]] = node_data
                elif graph_feature["domain"] == "edge":
                    edge_data = read_data(
                        os.path.join(dataset_dir, graph_feature["path"]),
                        graph_feature["format"],
                        in_memory=in_memory,
                    )
                    assert edge_data.shape[0] == num_edges
                    edge_attributes[graph_feature["name"]] = edge_data
        else:
            # Heterogeneous graph.
            node_feature_collector = {}
            edge_feature_collector = {}
            for graph_feature in graph_data["feature_data"]:
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
                    if graph_feature["name"] not in node_feature_collector:
                        node_feature_collector[graph_feature["name"]] = {}
                    node_feature_collector[graph_feature["name"]][
                        graph_feature["type"]
                    ] = node_data
                elif graph_feature["domain"] == "edge":
                    edge_data = read_data(
                        os.path.join(dataset_dir, graph_feature["path"]),
                        graph_feature["format"],
                        in_memory=in_memory,
                    )
                    if graph_feature["name"] not in edge_feature_collector:
                        edge_feature_collector[graph_feature["name"]] = {}
                    edge_feature_collector[graph_feature["name"]][
                        graph_feature["type"]
                    ] = edge_data

            # For heterogenous, a node/edge feature must cover all node/edge types.
            all_node_types = set(node_type_to_id.keys())
            for feat_name, feat_data in node_feature_collector.items():
                existing_node_type = set(feat_data.keys())
                assert all_node_types == existing_node_type, (
                    f"Node feature {feat_name} does not cover all node types. "
                    f"Existing types: {existing_node_type}. "
                    f"Expected types: {all_node_types}."
                )
            all_edge_types = set(edge_type_to_id.keys())
            for feat_name, feat_data in edge_feature_collector.items():
                existing_edge_type = set(feat_data.keys())
                assert all_edge_types == existing_edge_type, (
                    f"Edge feature {feat_name} does not cover all edge types. "
                    f"Existing types: {existing_edge_type}. "
                    f"Expected types: {all_edge_types}."
                )

            for feat_name, feat_data in node_feature_collector.items():
                _feat = next(iter(feat_data.values()))
                feat_tensor = torch.empty(
                    ([total_num_nodes] + list(_feat.shape[1:])),
                    dtype=_feat.dtype,
                )
                for ntype, feat in feat_data.items():
                    feat_tensor[
                        node_type_offset[
                            node_type_to_id[ntype]
                        ] : node_type_offset[node_type_to_id[ntype] + 1]
                    ] = feat
                node_attributes[feat_name] = feat_tensor
            del node_feature_collector
            for feat_name, feat_data in edge_feature_collector.items():
                _feat = next(iter(feat_data.values()))
                feat_tensor = torch.empty(
                    ([total_num_edges] + list(_feat.shape[1:])),
                    dtype=_feat.dtype,
                )
                for etype, feat in feat_data.items():
                    feat_tensor[
                        edge_type_offset[
                            edge_type_to_id[etype]
                        ] : edge_type_offset[edge_type_to_id[etype] + 1]
                    ] = feat
                edge_attributes[feat_name] = feat_tensor
            del edge_feature_collector

    if not bool(node_attributes):
        node_attributes = None
    if not bool(edge_attributes):
        edge_attributes = None

    # Construct the FusedCSCSamplingGraph.
    return fused_csc_sampling_graph(
        csc_indptr=indptr,
        indices=indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=node_type_to_id,
        edge_type_to_id=edge_type_to_id,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    )


def preprocess_ondisk_dataset(
    dataset_dir: str,
    include_original_edge_id: bool = False,
    force_preprocess: bool = None,
    auto_cast_to_optimal_dtype: bool = True,
) -> str:
    """Preprocess the on-disk dataset. Parse the input config file,
    load the data, and save the data in the format that GraphBolt supports.

    Parameters
    ----------
    dataset_dir : str
        The path to the dataset directory.
    include_original_edge_id : bool, optional
        Whether to include the original edge id in the FusedCSCSamplingGraph.
    force_preprocess: bool, optional
        Whether to force reload the ondisk dataset.
    auto_cast_to_optimal_dtype: bool, optional
        Casts the dtypes of tensors in the dataset into smallest possible dtypes
        for reduced storage requirements and potentially increased performance.
        Default is True.

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
    processed_dir_prefix = "preprocessed"
    preprocess_metadata_path = os.path.join(
        processed_dir_prefix, "metadata.yaml"
    )
    if os.path.exists(os.path.join(dataset_dir, preprocess_metadata_path)):
        if force_preprocess is None:
            with open(
                os.path.join(dataset_dir, preprocess_metadata_path), "r"
            ) as f:
                preprocess_config = yaml.safe_load(f)
            if (
                preprocess_config.get("include_original_edge_id", None)
                == include_original_edge_id
            ):
                force_preprocess = check_dataset_change(
                    dataset_dir, processed_dir_prefix
                )
            else:
                force_preprocess = True
        if force_preprocess:
            shutil.rmtree(os.path.join(dataset_dir, processed_dir_prefix))
            print(
                "The on-disk dataset is re-preprocessing, so the existing "
                + "preprocessed dataset has been removed."
            )
        else:
            print("The dataset is already preprocessed.")
            return os.path.join(dataset_dir, preprocess_metadata_path)

    print("Start to preprocess the on-disk dataset.")

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

    # 2. Load the data and create a FusedCSCSamplingGraph.
    if "graph" not in input_config:
        raise RuntimeError("Invalid config: does not contain graph field.")

    sampling_graph = _graph_data_to_fused_csc_sampling_graph(
        dataset_dir,
        input_config["graph"],
        include_original_edge_id,
        auto_cast_to_optimal_dtype,
    )

    # 3. Record value of include_original_edge_id.
    output_config["include_original_edge_id"] = include_original_edge_id

    # 4. Save the FusedCSCSamplingGraph and modify the output_config.
    output_config["graph_topology"] = {}
    output_config["graph_topology"]["type"] = "FusedCSCSamplingGraph"
    output_config["graph_topology"]["path"] = os.path.join(
        processed_dir_prefix, "fused_csc_sampling_graph.pt"
    )

    node_ids_within_int32 = (
        sampling_graph.indices.dtype == torch.int32
        and auto_cast_to_optimal_dtype
    )
    torch.save(
        sampling_graph,
        os.path.join(
            dataset_dir,
            output_config["graph_topology"]["path"],
        ),
    )
    del sampling_graph
    del output_config["graph"]

    # 5. Load the node/edge features and do necessary conversion.
    if input_config.get("feature_data", None):
        has_edge_feature_data = False
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
            if not has_edge_feature_data and feature["domain"] == "edge":
                has_edge_feature_data = True
            copy_or_convert_data(
                os.path.join(dataset_dir, feature["path"]),
                os.path.join(dataset_dir, out_feature["path"]),
                feature["format"],
                output_format=out_feature["format"],
                in_memory=in_memory,
                is_feature=True,
            )
        if has_edge_feature_data and not include_original_edge_id:
            gb_warning("Edge feature is stored, but edge IDs are not saved.")

    # 6. Save tasks and train/val/test split according to the output_config.
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
                        name = (
                            input_data["name"] if "name" in input_data else None
                        )
                        copy_or_convert_data(
                            os.path.join(dataset_dir, input_data["path"]),
                            os.path.join(dataset_dir, output_data["path"]),
                            input_data["format"],
                            output_data["format"],
                            within_int32=node_ids_within_int32
                            and name in NAMES_INDICATING_NODE_IDS,
                        )

    # 7. Save the output_config.
    output_config_path = os.path.join(dataset_dir, preprocess_metadata_path)
    with open(output_config_path, "w") as f:
        yaml.dump(output_config, f)
    print("Finish preprocessing the on-disk dataset.")

    # 8. Calculate and save the hash value of the dataset directory.
    hash_value_file = "dataset_hash_value.txt"
    hash_value_file_path = os.path.join(
        dataset_dir, processed_dir_prefix, hash_value_file
    )
    if os.path.exists(hash_value_file_path):
        os.remove(hash_value_file_path)
    dir_hash = calculate_dir_hash(dataset_dir)
    with open(hash_value_file_path, "w") as f:
        f.write(json.dumps(dir_hash, indent=4))

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
        train_set: Union[ItemSet, HeteroItemSet],
        validation_set: Union[ItemSet, HeteroItemSet],
        test_set: Union[ItemSet, HeteroItemSet],
    ):
        """Initialize a task.

        Parameters
        ----------
        metadata : Dict
            Metadata.
        train_set : Union[ItemSet, HeteroItemSet]
            Training set.
        validation_set : Union[ItemSet, HeteroItemSet]
            Validation set.
        test_set : Union[ItemSet, HeteroItemSet]
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
    def train_set(self) -> Union[ItemSet, HeteroItemSet]:
        """Return the training set."""
        return self._train_set

    @property
    def validation_set(self) -> Union[ItemSet, HeteroItemSet]:
        """Return the validation set."""
        return self._validation_set

    @property
    def test_set(self) -> Union[ItemSet, HeteroItemSet]:
        """Return the test set."""
        return self._test_set

    def __repr__(self) -> str:
        ret = "{Classname}({attributes})"

        attributes_str = ""

        attributes = get_attributes(self)
        attributes.reverse()
        for attribute in attributes:
            if attribute[0] == "_":
                continue
            value = getattr(self, attribute)
            attributes_str += f"{attribute}={value},\n"
        attributes_str = textwrap.indent(
            attributes_str, " " * len("OnDiskTask(")
        ).strip()

        return ret.format(
            Classname=self.__class__.__name__, attributes=attributes_str
        )


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
                  - name: seeds
                    format: numpy # Can be numpy or torch.
                    in_memory: true # If not specified, default to true.
                    path: set/paper-train-seeds.npy
                  - name: labels
                    format: numpy
                    path: set/paper-train-labels.npy
            validation_set:
              - type: paper
                data:
                  - name: seeds
                    format: numpy
                    path: set/paper-validation-seeds.npy
                  - name: labels
                    format: numpy
                    path: set/paper-validation-labels.npy
            test_set:
              - type: paper
                data:
                  - name: seeds
                    format: numpy
                    path: set/paper-test-seeds.npy
                  - name: labels
                    format: numpy
                    path: set/paper-test-labels.npy

    Parameters
    ----------
    path: str
        The YAML file path.
    include_original_edge_id: bool, optional
        Whether to include the original edge id in the FusedCSCSamplingGraph.
    force_preprocess: bool, optional
        Whether to force reload the ondisk dataset.
    auto_cast_to_optimal_dtype: bool, optional
        Casts the dtypes of tensors in the dataset into smallest possible dtypes
        for reduced storage requirements and potentially increased performance.
        Default is True.
    """

    def __init__(
        self,
        path: str,
        include_original_edge_id: bool = False,
        force_preprocess: bool = None,
        auto_cast_to_optimal_dtype: bool = True,
    ) -> None:
        # Always call the preprocess function first. If already preprocessed,
        # the function will return the original path directly.
        self._dataset_dir = path
        yaml_path = preprocess_ondisk_dataset(
            path,
            include_original_edge_id,
            force_preprocess,
            auto_cast_to_optimal_dtype,
        )
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

    def load(self, tasks: List[str] = None):
        """Load the dataset.

        Parameters
        ----------
        tasks: List[str] = None
            The name of the tasks to be loaded. For single task, the type of
            tasks can be both string and List[str]. For multiple tasks, only
            List[str] is acceptable.

        Examples
        --------
        1. Loading via single task name "node_classification".

        >>> dataset = gb.OnDiskDataset(base_dir).load(
        ...     tasks="node_classification")
        >>> len(dataset.tasks)
        1
        >>> dataset.tasks[0].metadata["name"]
        "node_classification"

        2. Loading via single task name ["node_classification"].

        >>> dataset = gb.OnDiskDataset(base_dir).load(
        ...     tasks=["node_classification"])
        >>> len(dataset.tasks)
        1
        >>> dataset.tasks[0].metadata["name"]
        "node_classification"

        3. Loading via multiple task names ["node_classification",
        "link_prediction"].

        >>> dataset = gb.OnDiskDataset(base_dir).load(
        ...     tasks=["node_classification","link_prediction"])
        >>> len(dataset.tasks)
        2
        >>> dataset.tasks[0].metadata["name"]
        "node_classification"
        >>> dataset.tasks[1].metadata["name"]
        "link_prediction"
        """
        self._convert_yaml_path_to_absolute_path()
        self._meta = OnDiskMetaData(**self._yaml_data)
        self._dataset_name = self._meta.dataset_name
        self._graph = self._load_graph(self._meta.graph_topology)
        self._feature = TorchBasedFeatureStore(self._meta.feature_data)
        self._tasks = self._init_tasks(self._meta.tasks, tasks)
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
    def all_nodes_set(self) -> Union[ItemSet, HeteroItemSet]:
        """Return the itemset containing all nodes."""
        self._check_loaded()
        return self._all_nodes_set

    def _init_tasks(
        self, tasks: List[OnDiskTaskData], selected_tasks: List[str]
    ) -> List[OnDiskTask]:
        """Initialize the tasks."""
        if isinstance(selected_tasks, str):
            selected_tasks = [selected_tasks]
        if selected_tasks and not isinstance(selected_tasks, list):
            raise TypeError(
                f"The type of selected_task should be list, but got {type(selected_tasks)}"
            )
        ret = []
        if tasks is None:
            return ret
        task_names = set()
        for task in tasks:
            task_name = task.extra_fields.get("name", None)
            if selected_tasks is None or task_name in selected_tasks:
                ret.append(
                    OnDiskTask(
                        task.extra_fields,
                        self._init_tvt_set(task.train_set),
                        self._init_tvt_set(task.validation_set),
                        self._init_tvt_set(task.test_set),
                    )
                )
                if selected_tasks:
                    task_names.add(task_name)
        if selected_tasks:
            not_found_tasks = set(selected_tasks) - task_names
            if len(not_found_tasks):
                gb_warning(
                    f"Below tasks are not found in YAML: {not_found_tasks}. Skipped."
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
    ) -> Union[ItemSet, HeteroItemSet]:
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
            itemsets = {}
            for tvt in tvt_set:
                itemsets[tvt.type] = ItemSet(
                    tuple(
                        read_data(data.path, data.format, data.in_memory)
                        for data in tvt.data
                    ),
                    names=tuple(data.name for data in tvt.data),
                )
            ret = HeteroItemSet(itemsets)
        return ret

    def _init_all_nodes_set(self, graph) -> Union[ItemSet, HeteroItemSet]:
        if graph is None:
            gb_warning(
                "`all_nodes_set` is returned as None, since graph is None."
            )
            return None
        num_nodes = graph.num_nodes
        dtype = graph.indices.dtype
        if isinstance(num_nodes, int):
            return ItemSet(
                torch.tensor(num_nodes, dtype=dtype),
                names="seeds",
            )
        else:
            data = {
                node_type: ItemSet(
                    torch.tensor(num_node, dtype=dtype),
                    names="seeds",
                )
                for node_type, num_node in num_nodes.items()
            }
            return HeteroItemSet(data)


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

    **ogbn-papers100M**
        The ogbn-papers100M dataset is a directed graph, representing the citation
        network between all Computer Science (CS) arXiv papers indexed by MAG.
        See more details in `ogbn-papers100M
        <https://ogb.stanford.edu/docs/nodeprop/#ogbn-papers100M>`_.

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

    **igb-hom and igb-hom-[tiny|small|medium|large]**
        The igb-hom-[tiny|small|medium|large] and igb-hom dataset is a homogeneous
        citation network, which is designed for developers to train and evaluate
        GNN models with high fidelity. See more details in
        `igb-hom-[tiny|small|medium|large]
        <https://github.com/IllinoisGraphBenchmark/IGB-Datasets>`_.

        .. note::
            Self edges are added to the original graph.
            Node features are stored as float32.

    **igb-het-[tiny|small|medium]**
        The igb-hom-[tiny|small|medium] dataset is a heterogeneous citation network,
        which is designed for developers to train and evaluate GNN models with
        high fidelity. See more details in `igb-het-[tiny|small|medium]
        <https://github.com/IllinoisGraphBenchmark/IGB-Datasets>`_.

        .. note::
            Four Reverse edge types are added to the original graph.
            Node features are stored as float32.

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
        "cora-seeds",
        "ogbn-mag",
        "ogbn-mag-seeds",
        "ogbl-citation2",
        "ogbl-citation2-seeds",
        "ogbn-products",
        "ogbn-products-seeds",
        "ogbn-arxiv",
        "ogbn-arxiv-seeds",
        "igb-hom-tiny",
        "igb-hom-tiny-seeds",
        "igb-hom-small",
        "igb-hom-small-seeds",
        "igb-het-tiny",
        "igb-het-tiny-seeds",
        "igb-het-small",
        "igb-het-small-seeds",
    ]
    _large_datasets = [
        "ogb-lsc-mag240m",
        "ogb-lsc-mag240m-seeds",
        "ogbn-papers100M",
        "ogbn-papers100M-seeds",
        "igb-hom-medium",
        "igb-hom-medium-seeds",
        "igb-hom-large",
        "igb-hom-large-seeds",
        "igb-hom",
        "igb-hom-seeds",
        "igb-het-medium",
        "igb-het-medium-seeds",
    ]
    _all_datasets = _datasets + _large_datasets

    def __init__(self, name: str, root: str = "datasets") -> OnDiskDataset:
        # For user using DGL 2.2 or later version, we prefer them to use
        # datasets with `seeds` suffix. This hack should be removed, when the
        # datasets with `seeds` suffix have covered previous ones.
        if "seeds" not in name:
            name += "-seeds"
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
        super().__init__(dataset_dir, force_preprocess=False)
