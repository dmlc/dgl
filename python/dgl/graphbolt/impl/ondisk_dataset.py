"""GraphBolt OnDiskDataset."""

import bisect
import json
import os
import pandas as pd
import shutil
import textwrap
from copy import deepcopy
from typing import Dict, List, Union
import time

import numpy as np

import torch
import yaml

from ...base import dgl_warning
from ...data.utils import download, extract_archive
from ..base import etype_str_to_tuple, ORIGINAL_EDGE_ID
from ..dataset import Dataset, Task
from ..internal import (
    calculate_dir_hash,
    check_dataset_change,
    copy_or_convert_data,
    get_attributes,
    read_data,
    read_edges,
)
from ..itemset import ItemSet, ItemSetDict
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
    "seed_nodes",
    "node_pairs",
    "seeds",
    "negative_srcs",
    "negative_dsts",
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
        print("!homogeneous graph")
        time1 = time.time()
        edge_fmt = graph_data["edges"][0]["format"]
        edge_path = graph_data["edges"][0]["path"]
        src, dst = read_edges(dataset_dir, edge_fmt, edge_path)
        num_nodes = graph_data["nodes"][0]["num"]
        num_edges = len(src)
        ##################################################
        # This part can we directly convert coo to csc
        # We just have src & dst, we do not have data
        # So for csc we just have indice & indptr
        # we need to sort coo by its column (probably the dst)
        # 1. Here we need the format of coo_tensor, a sample format
        # 2. Find a proper sort function
        # 3. consider something like bucket sort
        # src = [0,0,1,4,4,4,6]
        # dst = [0,2,2,2,3,4,3]
        # num_nodes = 7
        coo_tensor = torch.tensor(np.array([src, dst]))
        # print("coo_tensor: ", coo_tensor)
        sparse_matrix = spmatrix(coo_tensor, shape=(num_nodes, num_nodes))
        del coo_tensor
        indptr, indices, edge_ids = sparse_matrix.csc()
        del sparse_matrix
        time2 = time.time()
        print("total time: ", time2 - time1)
        # A sequential version of the algorithm
        indptr_re = [0]
        indice_re = []
        edge_ids_re = []
        # In outer sort mode
        # 1. first read in part of the csv and sort it 
        # 2. store the ordered result in the format (dst, src, original_idx) in temp csv
        # 3. we need to add an additional label for the original edge
        def sort_and_save_chunk(df, chunk_id, temp_dir):
            # sort the value in every csv
            df_sorted = df.sort_values(by=df.columns[1])
            # columns = df_sorted.columns.tolist()
            # 交换第一列和第二列的位置
            # columns[0], columns[1] = columns[1], columns[0]
            # 使用新的列顺序重新排列DataFrame
            # df_sorted = df_sorted[columns]
            # save the sorted csv
            sorted_chunk_path = os.path.join(temp_dir, f'sorted_chunk_{chunk_id}.npy')
            np_array = df_sorted.to_numpy()
            # print("np_array: ", np_array)
            np.save(sorted_chunk_path, np_array)
            return sorted_chunk_path

        temp_dir = 'temp_chunks'
        os.makedirs(temp_dir, exist_ok=True)
        chunk_id = 0
        paths = []
        # The unit for chunk_size is row number
        chunk_size = 1000 * 1000 * 20
        csv_path = "datasets/ogbn-products/edges/bi_edge.csv"
        # t1 = time.time()
        # pd.read_csv(csv_path, names=["src", "dst"])
        # t2 = time.time()
        # print("time for reading csv", t2-t1)
        time2 = time.time()
        # maybe can use some multi thread trick here, but when we use multi thread, do we reduce the memory consumption?
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, names=["src", "dst"]):
            t1 = time.time()
            sorted_chunk_path = sort_and_save_chunk(chunk, chunk_id, temp_dir)
            t2 = time.time()
            print("time for sort_and_save_chunk: ", t2-t1)
            paths.append(sorted_chunk_path)
            chunk_id += 1
        time3 = time.time()
        print("after split: ", time3 - time2)
        # currently this part is accelerated
        ##################################################
        # 3. merge the result in the temp csv
        # 3.1 read the result in temp csv with index
        # total_row = 0
        # for path in paths:
        #     sorted_chunk = pd.read_csv(path, names=["edge_id", "src", "dst"])
        #     total_row += len(sorted_chunk)

        # 4. the final result is the csv we need
        # 4.1 here is a lemma: if we read n lines from each chunk
        #     we can ensure that the top n element in the merged list is the smallest n
        #     elements in the final result.
        import heapq
        # 定义结果块的大小
        result_chunk_size = 10000 * 1000  # 举例，可以根据需要进行调整

        # 初始化一个列表来存储当前的结果块
        current_result_chunk = []
        arrays_iters = []
        heap = []
        np_array = []
        total_sz = 0
        # 定义保存结果的文件路径
        output_dir_path = temp_dir + "/"

        # 现在在归并排序过程中添加逻辑，以写入结果
        for file_idx, file_path in enumerate(paths):
            # 使用内存映射方式打开.npy文件
            # arr = np.lib.format.open_memmap(file_path, mode='r')
            # 这里我们按照文件大小创建迭代器，每次返回一小部分数组
            # array_iter = iter(np.array_split(arr, np.arange(chunk_size, arr.shape[0], chunk_size)))
            # array_iter = iter(arr)
            # array_iter = iter(np.load(file_path))
            # arrays_iters.append(array_iter)
            temp = np.load(file_path)
            np_array.append(temp)
            total_sz += temp.shape[0]
        # merged = heapq.merge(*arrays_iters, key=lambda x: x[-1])
        pq = []
        pointer = []
        xx = []
        # print(np_array)
        for i in range(len(np_array)):
            pointer.append(1)
            # numpy cannot directly push into pq, will case comparison error
            heapq.heappush(pq, (np_array[i][0][1], np_array[i][0].tolist(), i))
        cnt = 0
        chunk_idx = 0
        while(cnt < total_sz):
            _, next_item, idx = heapq.heappop(pq)
            current_result_chunk.append(next_item)
            cnt += 1
            if pointer[idx] < np_array[idx].shape[0]:
                # try:
                heapq.heappush(pq, (np_array[idx][pointer[idx]][1], np_array[idx][pointer[idx]].tolist(), idx))
                # except Exception as e:
                    # print("pointer idx", pointer[idx])

                pointer[idx] += 1
            if len(current_result_chunk) == result_chunk_size:
                with open(output_dir_path + str(chunk_idx) + "sorted.npy", 'wb') as f:
                    np.save(f, np.array(current_result_chunk))
                    chunk_idx += 1
                current_result_chunk = [] 
        # for row in merged:
        #     current_result_chunk.append(row)
            
        #     # 检查当前结果块是否达到指定大小
        #     if len(current_result_chunk) == result_chunk_size:
        #         # 将当前结果块写入文件
        #         with open(output_file_path, 'ab') as f:
        #             np.save(f, np.array(current_result_chunk))
        #         # 清空当前结果块，以便于存储新的结果
        #         current_result_chunk = []

        #     # 尝试从相同文件的迭代器中获取下一个元素
        #     try:
        #         next_item = next(arrays_iters[file_idx])
        #     except StopIteration:
        #         continue

        # 确保所有剩余数据都写入文件
        if current_result_chunk:
            with open(output_dir_path + str(chunk_idx) + "sorted.npy", 'wb') as f:
                np.save(f, np.array(current_result_chunk))
                chunk_idx += 1
        print("cnt: ", cnt)
        print("chunk_idx", chunk_idx)
        # data = np.load(output_file_path)
        # print(data, data.shape)
        print("merge finish!")
        # 5. now we change the format of coo to a csc changeable one
        #    Then we perform the loop before to construct a csc graph
        last_col = -1
        i = 0
        time4 = time.time()
        print("before convert to csc: ", time4 - time2)

        # 按块读取和迭代数据
        for i in range(chunk_idx):
            # 获取当前块的数据
            chunk = np.load(output_dir_path + str(i) + "sorted.npy")
            print("chunk: ", chunk)
            # 迭代当前块中的每一行
            for xx in chunk:
                # 在这里处理每一行，xx是当前行的数据
                # edge_id = xx[0]
                # row = xx[1]
                # col = xx[2]
                row = xx[0]
                col = xx[1]
                if i >= 1 and col > last_col:
                    for j in range(last_col + 1, col + 1):
                        indptr_re.append(i)
                indice_re.append(row)
                # edge_ids_re.append(edge_id)
                i += 1
                last_col = col
        # 6. remove the temp file
        for path in paths:
            os.remove(path)
            # os.rmdir(temp_dir)
        return 
        # src_npy = np.array(src)
        # dst_npy = np.array(dst)
        # new_indices = np.argsort(dst_npy)
        # sorted_src = src_npy[new_indices]
        # sorted_dst = dst_npy[new_indices]
        # for i in range(len(sorted_dst)):
        #     if i >= 1 and sorted_dst[i] > sorted_dst[i-1]:
        #         for j in range(sorted_dst[i-1] + 1, sorted_dst[i] + 1):
        #             indptr_re.append(i)
        #     indice_re.append(sorted_src[i])
        #     edge_ids_re.append(new_indices[i])
        # the upper bound
        while(len(indptr_re) <= num_nodes):
            indptr_re.append(len(dst))
        
        indptr_re = torch.tensor(indptr_re)
        indice_re = torch.tensor(indice_re)
        edge_ids_re = torch.tensor(edge_ids_re)
        time5 = time.time()
        print("total time 2: ", time5 - time2)
        print("indptr: ", indptr, indptr.size())
        print("indptr_re: ", indptr_re, indptr_re.size())
        print("indices: ", indices, indices.size())
        print("indice_re: ", indice_re, indice_re.size())
        print("edge id: ", edge_ids)
        print("edge_ids_re", edge_ids_re)
       
        # 6.0 verification
        # a = []
        # b = []
        # for i in range(174):
        #     a.append([indices[i], edge_ids[i]])
        #     b.append([indice_re[i], edge_ids_re[i]])
        # a.sort()
        # b.sort()
        # print("a:", a[:10])
        # print("b:", b[:10])

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
            dgl_warning("Edge feature is stored, but edge IDs are not saved.")

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
    def all_nodes_set(self) -> Union[ItemSet, ItemSetDict]:
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
                dgl_warning(
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
                "`all_nodes_set` is returned as None, since graph is None."
            )
            return None
        num_nodes = graph.num_nodes
        dtype = graph.indices.dtype
        if isinstance(num_nodes, int):
            return ItemSet(
                torch.tensor(num_nodes, dtype=dtype),
                names="seed_nodes",
            )
        else:
            data = {
                node_type: ItemSet(
                    torch.tensor(num_node, dtype=dtype),
                    names="seed_nodes",
                )
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
    _large_datasets = ["ogb-lsc-mag240m", "ogbn-papers100M"]
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
        super().__init__(dataset_dir, force_preprocess=False)
