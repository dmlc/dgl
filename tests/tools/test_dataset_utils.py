import os
import tempfile
from datetime import timedelta

import numpy as np
import pytest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from distpartitioning import array_readwriter, constants

from distpartitioning.dataset_utils import get_dataset
from distpartitioning.utils import generate_read_list


def _validate_edges(
    rank, world_size, num_chunks, edge_dict, edge_tids, schema_map
):
    """Function to check the correctness of the edges

    Parameters:
    -----------
    rank : int
        id of the process
    world_size : int
        no. of processes to spawn
    num_chunks : int
        no. of files to create for edge, node and edge features
    edge_tids : dict[str, list]
        type-id range of edges read from the disk
    schema_map : dict[str, list/str/dict]
        dictionary object to describe the input dataset
    """
    # validate edge data here for this rank
    read_files = generate_read_list(num_chunks, world_size)
    exp_src_ids = []
    exp_dst_ids = []
    exp_etype_ids = []
    for idx in read_files[rank]:
        data = (
            np.array([np.arange(10), np.arange(10)]).reshape(10, 2) + 10 * idx
        )
        exp_src_ids.append(data[:, 0])
        exp_dst_ids.append(data[:, 1])
        exp_etype_ids.append(np.zeros((data.shape[0],), dtype=np.int64))

    if len(exp_src_ids) == 0:
        assert (len(edge_tids["n1:e1:n1"]) - 1) < rank
    else:
        exp_src_ids = np.concatenate(exp_src_ids)
        exp_dst_ids = np.concatenate(exp_dst_ids)
        exp_etype_ids = np.concatenate(exp_etype_ids)

        assert np.all(exp_src_ids == edge_dict[constants.GLOBAL_SRC_ID])
        assert np.all(exp_dst_ids == edge_dict[constants.GLOBAL_DST_ID])
        assert np.all(exp_etype_ids == edge_dict[constants.ETYPE_ID])

        # validate edge_tids here.
        assert edge_tids["n1:e1:n1"][0] == (
            rank * num_chunks * 10,
            (rank + 1) * num_chunks * 10,
        )


def _validate_edge_data(
    rank, world_size, num_chunks, edge_features, edge_feature_tids
):
    """Function to check the correctness of the edge features

    Parameters:
    -----------
    rank : int
        id of the process
    world_size : int
        no. of processes to spawn
    num_chunks : int
        no. of files to create for edge, node and edge features
    edge_features : dict[str, Tensor]
        edge features that are read from the disk
    edge_feature_tids : dict[str, list]
        type-id range of edge features read from the disk
    """
    # validate edge feat here.
    read_files = generate_read_list(num_chunks, world_size)
    edge_feats = []
    for idx in read_files[rank]:
        data = np.arange(10, dtype=np.int32)
        for _ in range(9):
            data = np.vstack((data, np.arange(10, dtype=np.int32) + 10 * idx))
        edge_feats.append(data)

    if len(edge_feats) == 0:
        actual_results = edge_feats["n1:e1:n1/edge_feat_1/0"]
        assert actual_results == None
        assert edge_feature_tids["n1:e1:n1/edge_feat_1/0"][0] == (0, 0)
    else:
        edge_feats = np.concatenate(edge_feats)

        # assert
        assert np.all(
            edge_feats == edge_features["n1:e1:n1/edge_feat_1/0"].numpy()
        )
        assert edge_feature_tids["n1:e1:n1/edge_feat_1/0"][0] == (0, 10)


def _validate_node_data(
    rank, world_size, num_chunks, node_features, node_feature_tids
):
    """Function to check the correctness of the node features

    Parameters:
    -----------
    rank : int
        id of the process
    world_size : int
        no. of processes to spawn
    num_chunks : int
        no. of files to create for edge, node and edge features
    node_features : dict[str, Tensor]
        node features that are read from the disk
    node_feature_tids : dict[str, list]
        type-id range of node features read from the disk
    """
    # validate node feat here.
    read_files = generate_read_list(num_chunks, world_size)
    node_feats = []
    for idx in read_files[rank]:
        data = np.arange(100, 110, dtype=np.int64)
        for _ in range(9):
            data = np.vstack(
                (data, np.arange(100, 110, dtype=np.int64) + 100 * idx)
            )
        node_feats.append(data)

    if len(node_feats) == 0:
        actual_results = node_features["n1/node_feat_1/0"]
        assert actual_results == None
        assert node_feature_tids["n1/node_feat_1/0"][0] == (0, 0)
    else:
        node_feats = np.concatenate(node_feats)

        # assert
        assert np.all(node_feats == node_features["n1/node_feat_1/0"].numpy())
        assert node_feature_tids["n1/node_feat_1/0"][0] == (0, 10)


def _get_test_data(
    node_feat_dir,
    edges_dir,
    edge_feat_dir,
    num_chunks,
    edge_fmt="csv",
    edge_fmt_del=" ",
):
    """Create a test graph for testing purpose

    Parameters:
    ----------
    node_feat_dir : str
        location to store the node feature files
    edges_dir : str
        location to stores files for edges
    edge_feat_dir : str
        location to store the edge feaature files
    num_chunks : int
        no. of files to create for edge and node and edge feature files
    edge_fmt : str
        format of the file in which edges are stored
    edge_fmt_del : str
        seperator to use when storing the edges in a file
    """
    schema = {}
    schema["num_nodes_per_type"] = [10]
    schema["edge_type"] = ["n1:e1:n1"]
    schema["node_type"] = ["n1"]

    edges = {}
    edges["n1:e1:n1"] = {}
    edges["n1:e1:n1"]["format"] = {}
    edges["n1:e1:n1"]["format"]["name"] = edge_fmt
    edges["n1:e1:n1"]["format"]["delimiter"] = edge_fmt_del

    os.makedirs(edges_dir, exist_ok=True)
    fmt_meta = {"name": edge_fmt}
    if edge_fmt == "csv":
        fmt_meta["delimiter"] = edge_fmt_del

    for idx in range(num_chunks):
        path = os.path.join(edges_dir, f"test_file_{idx}.{fmt_meta['name']}")
        array_parser = array_readwriter.get_array_parser(**fmt_meta)
        edge_data = (
            np.array([np.arange(10), np.arange(10)]).reshape(10, 2) + 10 * idx
        )
        array_parser.write(path, edge_data)

    edge_files = [path]
    edges["n1:e1:n1"]["data"] = edge_files
    schema["edges"] = edges

    schema["edge_data"] = {}
    schema["edge_data"]["n1:e1:n1"] = {}
    schema["edge_data"]["n1:e1:n1"]["edge_feat_1"] = {}
    schema["edge_data"]["n1:e1:n1"]["edge_feat_1"]["format"] = {}
    schema["edge_data"]["n1:e1:n1"]["edge_feat_1"]["format"]["name"] = "numpy"

    edge_feat_files = []
    os.makedirs(edge_feat_dir, exist_ok=True)
    for idx in range(num_chunks):
        path = os.path.join(edge_feat_dir, f"test_edge_feat_{idx}.npy")
        data = np.arange(10, dtype=np.int32)
        for _ in range(9):
            data = np.vstack((data, np.arange(10, dtype=np.int32) + 10 * idx))
        np.save(path, data)
        edge_feat_files.append(path)
    schema["edge_data"]["n1:e1:n1"]["edge_feat_1"]["data"] = edge_feat_files

    schema["node_data"] = {}
    schema["node_data"]["n1"] = {}
    schema["node_data"]["n1"]["node_feat_1"] = {}
    schema["node_data"]["n1"]["node_feat_1"]["format"] = {}
    schema["node_data"]["n1"]["node_feat_1"]["format"]["name"] = "numpy"
    node_feat_files = []
    os.makedirs(node_feat_dir, exist_ok=True)
    for idx in range(num_chunks):
        path = os.path.join(node_feat_dir, f"test_node_feat_{idx}.npy")
        data = np.arange(100, 110, dtype=np.int64)
        for _ in range(9):
            data = np.vstack(
                (data, np.arange(100, 110, dtype=np.int64) + 100 * idx)
            )
        np.save(path, data)
        node_feat_files.append(path)
    schema["node_data"]["n1"]["node_feat_1"]["data"] = node_feat_files

    return schema


def _init_process_group(rank, world_size):
    """Function to init the process group for communication
    Parameters:
    -----------
    rank : int
        id of the process
    world_size : int
        no. of processes to spawn
    """
    # init the gloo process group here.
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )


def _run(
    port_num,
    rank,
    world_size,
    num_chunks,
    num_parts,
    sh_dict,
    schema_map,
    input_dir,
):
    """Main function for each spawned process, mimicing the actual process when
    the pipeline is executed

    Parameters:
    -----------
    port_num : int
        port to use for communication
    rank : int
        id of the process
    world_size : int
        no. of processes to spawn
    num_chunks : int
        no. of files to create for edge, node and edge features
    num_parts : int
        no. of output graph partitions
    sh_dict : dict[str, str]
        shared dictionary to pass error strings to the parent process
    schema_map : dict[str, list/str/dict]
        dictionary to describe the input dataset
    input_dir : str
        location where input dataset is stored
    """
    try:

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port_num)
        _init_process_group(rank, world_size)

        ntype_counts = dict(
            zip(schema_map["node_type"], schema_map["num_nodes_per_type"])
        )

        (
            node_features,
            node_feature_tids,
            edge_datadict,
            edge_typecounts,
            edge_tids,
            edge_features,
            edge_feature_tids,
        ) = get_dataset(
            input_dir,
            "test_graph",
            rank,
            world_size,
            num_parts,
            schema_map,
            ntype_counts,
        )

        _validate_node_data(
            rank, world_size, num_chunks, node_features, node_feature_tids
        )
        _validate_edge_data(
            rank, world_size, num_chunks, edge_features, edge_feature_tids
        )
        _validate_edges(
            rank, world_size, num_chunks, edge_datadict, edge_tids, schema_map
        )

    except Exception as arg:
        sh_dict[f"RANK-{rank}"] = arg


def _single_machine_run(
    world_size, num_chunks, num_parts, schema_map, input_dir
):
    """Auxiliary function to spawn processes and gather errors if any

    Parameters:
    -----------
    world_size : int
        no. of processes to spawn
    num_chunks : int
        no. of files to create for edge, node and edge features
    num_parts : int
        no. of output graph partitions
    schema_map : dict[str, list/str/dict]
        dictionary to describe the input dataset
    input_dir : str
        location where input dataset is stored
    """
    port_num = np.random.randint(10000, 20000, size=(1,), dtype=int)[0]
    ctx = mp.get_context("spawn")
    manager = mp.Manager()

    # shared dictionary to store any assertion failures in spawned processes
    sh_dict = manager.dict()

    # spawn processes to fire the unit test cases
    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_run,
            args=(
                port_num,
                rank,
                world_size,
                num_chunks,
                num_parts,
                sh_dict,
                schema_map,
                input_dir,
            ),
        )
        p.start()
        processes.append(p)

    # wait for the processes to join
    for p in processes:
        p.join()
        p.close()

    # Make sure that the spawned process, mimicing ranks/workers, did
    # not generate any errors or assertion failures
    assert len(sh_dict) == 0, f"Spawned processes reported some errors !!!"


@pytest.mark.parametrize(
    "world_size, num_chunks, num_parts", [[1, 1, 4], [4, 1, 4]]
)
@pytest.mark.parametrize("edges_fmt", ["csv", "parquet"])
@pytest.mark.parametrize("edges_delimiter", [" ", ","])
def test_get_dataset(
    world_size, num_chunks, num_parts, edges_fmt, edges_delimiter
):
    """Unit tests for testing reading the dataset from the disk

    Parameters:
    -----------
    world_size : int
        no. of processes to spawn
    num_chunks : int
        no of files to create for edges, node and edge features
    num_parts : int
        no. of output graph partitions
    edges_fmt : str
        format in which to store the edges for the input graph
    edges_delimiter : str
        delimiter to use for edges files
    """
    # Create the input dataset
    with tempfile.TemporaryDirectory() as root_dir:

        # Prepare the state information for firing unit test
        input_dir = os.path.join(root_dir, "chunked-data")
        edge_dir = os.path.join(input_dir, "edges")
        edge_feat_dir = os.path.join(input_dir, "edge_feat")
        node_feat_dir = os.path.join(input_dir, "node_feat")
        output_dir = os.path.join(root_dir, "preproc_dir")

        # Read the input schema
        schema_map = _get_test_data(
            node_feat_dir,
            edge_dir,
            edge_feat_dir,
            num_chunks,
            edges_fmt,
            edges_delimiter,
        )

        # fire the unit test case.
        _single_machine_run(
            world_size, num_chunks, num_parts, schema_map, input_dir
        )
