import os
import tempfile
from datetime import timedelta

import dgl
import numpy as np
import pytest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from distpartitioning import array_readwriter, constants
from distpartitioning.data_shuffle import exchange_features
from distpartitioning.dist_lookup import DistLookupService

from distpartitioning.utils import DATA_TYPE_ID

"""
DATA_TYPE_ID = {
    data_type: id
    for id, data_type in enumerate(
        [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
        ]
    )
}
"""

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def _get_type_id_ranges(
    data, rank, world_size, num_parts, num_chunks, feat_name
):
    """Function to generate type-id ranges for node/edge features

    Parameters:
    ----------
    data : numpy array
        Features on any given rank
    rank : int
        rank of the current process
    world_size : int
        no. of participating processes
    num_parts : int
        no. of graph partitions
    num_chunks : int
        no. of files/chunks to generate for edges and node/edge features
    feat_name : str
        name of the feature, which is used as a key in the dictionary

    Returns:
    --------
    dict :
        where keys are feature names and values are list of tuples.
        No. of tuples is equal to the total no. of processes.
    """
    data_shape = list(data.shape)
    if len(data_shape) == 1:
        data_shape.append(1)

    data_shape.append(DATA_TYPE_ID[data.dtype])

    data_shape = torch.tensor(data_shape, dtype=torch.int64)
    data_shape_output = [
        torch.zeros_like(data_shape) for _ in range(world_size)
    ]
    dist.all_gather(data_shape_output, data_shape)

    shapes = [x.numpy() for x in data_shape_output if x[0] != 0]
    shapes = np.vstack(shapes)

    type_counts = list(shapes[:, 0])
    tid_start = np.cumsum([0] + type_counts[:-1])
    tid_end = np.cumsum(type_counts)
    tid_ranges = list(zip(tid_start, tid_end))

    return tid_ranges


def _init_process_group(rank, world_size):
    """Function to init process group

    Parameters:
    -----------
    rank : int
        rank of the current process
    world_size : int
        no. of participating processes
    """
    # init the gloo process group here.
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )


def _get_edges(input_dir, rank, num_chunks, num_parts, world_size, schema_map):
    """Function to create data structures which are the function arguments
    for the `exchange_features` function.

    Parameters:
    -----------
    input_dir : str
        directory where the input dataset is stored
    rank : int
        rank of the current process
    num_chunks : int
        no. of files/chunks to generate for edges and node/edge features
    num_parts : int
        no. of graph partitions
    world_size : int
        no. of participating processes
    schema_map : dict
        describing the metadata of the unit test graph

    Returns:
    --------
    dict :
        where keys are the edge attributes and values are numpy arrays.
        This dictionary is the key input parameter to the
        `exchange_features` function.
    """
    # read edge feats
    edge_data = {}
    edge_data[constants.GLOBAL_SRC_ID] = []
    edge_data[constants.GLOBAL_DST_ID] = []
    edge_data[constants.GLOBAL_TYPE_EID] = []
    edge_data[constants.ETYPE_ID] = []
    edge_data[constants.GLOBAL_EID] = []

    etypes = schema_map[constants.STR_EDGE_TYPE]
    etype_etypeid_map = {e: i for i, e in enumerate(etypes)}

    input_edges = schema_map["edges"]
    for etype_name, etype_data in input_edges.items():
        num_files = len(etype_data["data"])
        fmt_meta = {"name": etype_data["format"]["name"], "delimiter": " "}
        data = []
        for fname in etype_data["data"]:
            data.append(
                array_readwriter.get_array_parser(**fmt_meta).read(fname)
            )
        data = np.concatenate(data)

        start = 0
        end = num_chunks * 10
        edge_data[constants.GLOBAL_SRC_ID].append(data[:, 0].astype(np.int64))
        edge_data[constants.GLOBAL_DST_ID].append(data[:, 1].astype(np.int64))
        edge_data[constants.GLOBAL_TYPE_EID].append(
            np.arange(start, end, dtype=np.int64)
        )
        edge_data[constants.ETYPE_ID].append(
            np.ones((data.shape[0],), dtype=np.int64)
            * etype_etypeid_map[etype_name]
        )
        edge_data[constants.GLOBAL_EID].append(
            np.arange(start, end, dtype=np.int64)
        )

    edge_data[constants.GLOBAL_SRC_ID] = np.concatenate(
        edge_data[constants.GLOBAL_SRC_ID]
    )
    edge_data[constants.GLOBAL_DST_ID] = np.concatenate(
        edge_data[constants.GLOBAL_DST_ID]
    )
    edge_data[constants.GLOBAL_TYPE_EID] = np.concatenate(
        edge_data[constants.GLOBAL_TYPE_EID]
    )
    edge_data[constants.ETYPE_ID] = np.concatenate(
        edge_data[constants.ETYPE_ID]
    )
    edge_data[constants.GLOBAL_EID] = np.concatenate(
        edge_data[constants.GLOBAL_EID]
    )

    return edge_data


def _read_edge_feats(
    input_dir, rank, num_chunks, num_parts, world_size, schema_map
):
    """Function to read edge feature from the disk.

    Parameters:
    -----------
    input_dir : str
        directory where the input dataset is stored
    rank : int
        rank of the current process
    num_chunks : int
        no. of files/chunks to generate for edges and node/edge features
    num_parts : int
        no. of graph partitions
    world_size : int
        no. of participating processes
    schema_map : dict
        describing the metadata of the unit test graph

    Returns:
    -------
    dict :
        used to store edge features, after reading from the disk. Keys are
        edge feature names and values are numpy arrays
    dict :
        used to store type-ids for the edge features after reading from the
        disk. Keys are edge feature names and value is a list which has
        one tuple in it. This tuple indicates the starting and ending
        type-id of the edge-features for this edge-type.
    dict :
        used to store global edge-id range for any give edge-type in the
        input graph
    """
    edge_feats = {}
    edge_feat_tids = {}

    # geid offset
    etype_geid_offset = {}
    etype_geid_offset["n1:e1:n1"] = [0, 10 * num_chunks]
    etype_geid_offset["n1:rev-e1:n1"] = [10 * num_chunks, 2 * 10 * num_chunks]

    # read edge feats
    input_edge_feats = schema_map["edge_data"]
    for etype_name, etype_feat_data in input_edge_feats.items():
        for feat_name, feat_data in etype_feat_data.items():
            num_files = len(feat_data["data"])
            read_list = np.split(np.arange(num_files), world_size)
            fmt_meta = {"name": feat_data["format"]["name"]}
            data = []
            for idx in read_list[rank]:
                fname = feat_data["data"][idx]
                data.append(
                    array_readwriter.get_array_parser(**fmt_meta).read(fname)
                )

            if len(data) > 0:
                data = np.concatenate(data)
            else:
                data = np.array([])

            data = torch.from_numpy(data)
            type_ids = _get_type_id_ranges(
                data,
                rank,
                world_size,
                num_parts,
                num_chunks,
                f"{etype_name}/{feat_name}",
            )
            data_key = f"{etype_name}/{feat_name}/0"
            if len(type_ids) > rank:
                start, end = type_ids[rank]
                edge_feats[data_key] = data
                edge_feat_tids[data_key] = [(start, end)]
            else:
                edge_feats[data_key] = None
                edge_feat_tids[data_key] = [(0, 0)]
    return edge_feats, edge_feat_tids, etype_geid_offset


def _read_node_feats(
    input_dir, rank, num_chunks, num_parts, world_size, schema_map
):
    """Function to read node feature from the disk.

    Parameters:
    -----------
    input_dir : str
        directory where the input dataset is stored
    rank : int
        rank of the current process
    num_chunks : int
        no. of files/chunks to generate for edges and node/edge features
    num_parts : int
        no. of graph partitions
    world_size : int
        no. of participating processes
    schema_map : dict
        describing the metadata of the unit test graph

    Returns:
    -------
    dict :
        used to store node features, after reading from the disk. Keys are
        node feature names and values are numpy arrays
    dict :
        used to store type-ids for the node features after reading from the
        disk. Keys are node feature names and value is a list which has
        one tuple in it. This tuple indicates the starting and ending
        type-id of the node-features for this node-type.
    dict :
        used to store global node-id range for any give node-type in the
        input graph
    """
    node_feats = {}
    node_feat_tids = {}

    # gnid offset
    ntype_gnid_offset = {}
    ntype_gnid_offset["n1"] = [0, 10 * num_chunks]

    # read node feats
    input_node_feats = schema_map["node_data"]
    for ntype_name, ntype_feat_data in input_node_feats.items():
        for feat_name, feat_data in ntype_feat_data.items():
            num_files = len(feat_data["data"])
            read_list = np.split(np.arange(num_files), world_size)
            fmt_meta = {"name": feat_data["format"]["name"]}
            data = []
            for idx in read_list[rank]:
                fname = feat_data["data"][idx]
                data.append(
                    array_readwriter.get_array_parser(**fmt_meta).read(fname)
                )

            if len(data) > 0:
                data = np.concatenate(data)
            else:
                data = np.array([])

            data = torch.from_numpy(data)

            type_ids = _get_type_id_ranges(
                data,
                rank,
                world_size,
                num_parts,
                num_chunks,
                f"{ntype_name}/{feat_name}",
            )
            data_key = f"{ntype_name}/{feat_name}/0"
            if len(type_ids) > rank:
                start, end = type_ids[rank]
                node_feats[data_key] = data
                node_feat_tids[data_key] = [(start, end)]
            else:
                node_feats[data_key] = None
                node_feat_tids[data_key] = [(0, 0)]

    return node_feats, node_feat_tids, ntype_gnid_offset


def _gen_expected_feats(num_chunks, feat_dim, feat_dtype):
    """Function to generate features, expected, for validating the test case

    Parameters:
    -----------
    num_chunks : int
        no. of files/chunks to generate for edges and node/edge features
    feat_dim : int
        no. of dimensions of the features. For instance, in a 2x2 matrix
        of size x rows and y columns `y` refers to feature dimensions
    feat_dtype : numpy dtype
        numpy dtype used when generating features
    """
    # node ownership is round-robin
    # node feats are generated as follows
    expected_data = []
    for idx in range(num_chunks):
        data = np.arange(feat_dim).astype(feat_dtype)
        for t in range(1, 10):
            arr = np.arange(t, t + feat_dim).astype(feat_dtype) + idx * 10
            data = np.vstack((data, arr))
        data = data.astype(feat_dtype)
        expected_data.append(data)
    expected_data = np.concatenate(expected_data)
    return expected_data


def _validate_shuffled_efeats(
    rank,
    world_size,
    num_chunks,
    num_parts,
    feats,
    global_ids,
    edge_feat_dim,
    edge_feat_dtype,
    edge_data,
):
    """Function used to compare expected and actual results for edge features

    Parameters:
    -----------
    rank : int
        rank of the current process
    world_size : int
        no. of participating processes
    num_chunks : int
        no. of files/chunks to generate for edges and node/edge features
    num_parts : int
        no. of graph partitions
    feats : dict
        where the key is the node feature name and value are the edge feature
        tensors after shuffling in the current process
    global_ids : int, tensor
        list of integers representing the global (dst) node-ids for the edge
        features after shuffling in the current process
    edge_feat_dim : int
        no. of dimensions for node features. For instance, in a 2x2 matrix
        of size x rows and y columns `y` refers to feature dimensions
    edge_feat_dtype : numpy dtype
        numpy dtype used when generating node features
    edge_data : dict
        where keys are the columns for edge attributes and values are numpy
        arrays
    """
    # Data to compare
    shuffled_efeats = feats["n1:e1:n1/edge_feat_1/0"].numpy()
    shuffled_global_eids = global_ids["n1:e1:n1/edge_feat_1/0"].numpy()

    sorted_idx = np.argsort(shuffled_global_eids)
    shuffled_global_eids = shuffled_global_eids[sorted_idx]
    shuffled_efeats = shuffled_efeats[sorted_idx]

    # Data for this rank
    dst_ids = edge_data[constants.GLOBAL_DST_ID]
    global_eids = edge_data[constants.GLOBAL_EID]
    edge_types = edge_data[constants.ETYPE_ID]

    idxes = np.where(edge_types == 0)[0]
    dst_ids = dst_ids[idxes]
    global_eids = global_eids[idxes]

    # global node ids
    num_nodes = 10 * num_chunks
    node_ids = np.arange(num_nodes)

    # partition ids
    num_repeats = np.ceil(num_nodes / num_parts).astype(np.int64)
    part_ids = np.tile(np.arange(num_parts), num_repeats)[:num_nodes]

    # condition to map partition-ids to ranks/workers
    dst_owners = part_ids[dst_ids]
    condition = dst_owners == rank
    idxes = np.where(condition == 1)[0]
    local_edge_ids = global_eids[idxes]

    # expected edge feats
    expected_data = _gen_expected_feats(
        num_chunks, edge_feat_dim, edge_feat_dtype
    )

    sorted_idx = np.argsort(local_edge_ids)
    expected_feats = expected_data[local_edge_ids]

    # assert on the expected and actual results
    assert np.all(shuffled_efeats == expected_feats)


def _validate_shuffled_nfeats(
    rank,
    world_size,
    num_chunks,
    feats,
    global_ids,
    node_feat_dim,
    node_feat_dtype,
):
    """Function used to compare expected and actual results for node features

    Parameters:
    -----------
    rank : int
        rank of the current process
    world_size : int
        no. of participating processes
    num_chunks : int
        no. of files/chunks to generate for edges and node/edge features
    feats : dict
        where the key is the node feature name and value are the node feature
        tensors after shuffling in the current process
    global_ids : int, tensor
        list of integers representing the global node-ids for the node
        features after shuffling in the current process
    node_feat_dim : int
        no. of dimensions for node features. For instance, in a 2x2 matrix
        of size x rows and y columns `y` refers to feature dimensions
    node_feat_dtype : numpy dtype
        numpy dtype used when generating node features
    """
    # tensors to compare
    nfeats = feats["n1/node_feat_1/0"]
    expected_data = _gen_expected_feats(
        num_chunks, node_feat_dim, node_feat_dtype
    )

    # get the data for this rank
    rowids = []
    offset = rank
    for idx in range(10 * num_chunks):
        if offset >= 10 * num_chunks:
            break
        rowids.append(offset)
        offset += world_size

    # ranks' data
    expected_data = expected_data[rowids, :]
    assert np.all(expected_data == nfeats.numpy())


def _run(
    rank,
    world_size,
    num_parts,
    num_chunks,
    schema_map,
    partitions_dir,
    data_dir,
    port_num,
    feat_mesg_size,
    sh_dict,
    node_feat_dim,
    node_feat_dtype,
    edge_feat_dim,
    edge_feat_dtype,
    feat_type,
):
    """Main function to be executed by the spawned process for unit testing
    purposes

    Parameters:
    -----------
    rank : int
        rank of the current process
    world_size : int
        no. of participating processes
    num_parts : int
        no. of graph partitions
    num_chunks : int
        no. of files/chunks to generate for edges and node/edge features
    schema_map : dict
        describing the metadata of the unit test graph
    partitions_dir : str
        directory where the node-id to partition-id mappings are located
    data_dir : str
        directory where the dataset is stored
    port_num : int
        port number used for communication by the process group
    feat_mesg_size : int
        maximum size of the outgoing message used by ``exchange_features``
        function when shuffling node/edge features
    sh_dict : dictionary
        shared dictionary for reporting errors back to the parent process
    node_feat_dim : int
        no. of dimensions for node features. For instance, in a 2x2 matrix
        of size x rows and y columns `y` refers to feature dimensions
    node_feat_dtype : numpy dtype
        numpy dtype used when generating node features
    edge_feat_dim : int
        no. of dimensions for edge features (similar to node_feat_dim above)
    edge_feat_dtype : int
        numpy dtype used when generating edge features
    feat_type : string
        value is either `node_features` or `edge_features`
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port_num)

    # pg init
    _init_process_group(rank, world_size)

    # look up service
    id_lookup = DistLookupService(
        partitions_dir,
        schema_map["node_type"],
        rank,
        world_size,
        num_parts,
    )

    # read dataset
    features = None
    feature_tids = None
    type_gid_rangemap = None
    edge_data = None
    if feat_type == "node_features":
        features, feature_tids, type_gid_rangemap = _read_node_feats(
            data_dir, rank, num_chunks, num_parts, world_size, schema_map
        )
    else:
        features, feature_tids, type_gid_rangemap = _read_edge_feats(
            data_dir, rank, num_chunks, num_parts, world_size, schema_map
        )
        edge_data = _get_edges(
            data_dir, rank, num_chunks, num_parts, world_size, schema_map
        )

    # set id_map in dist_lookup service
    ntype_gnid_rangemap = {}
    ntype_gnid_rangemap["n1"] = [0, 10 * num_chunks]

    global_nid_ranges = {}
    global_nid_ranges["n1"] = np.array(ntype_gnid_rangemap["n1"]).reshape(
        [1, 2]
    )
    id_map = dgl.distributed.id_map.IdMap(global_nid_ranges)
    id_lookup.set_idMap(id_map)

    # fire the test case.
    try:
        if feat_type == "node_features":
            my_feats, my_global_ids = exchange_features(
                rank,
                world_size,
                num_parts,
                feat_mesg_size,
                feature_tids,
                ntype_gid_rangemap,
                id_lookup,
                features,
                feat_type,
                edge_data,
            )
        else:
            my_feats, my_global_ids = exchange_features(
                rank,
                world_size,
                num_parts,
                feat_mesg_size,
                feature_tids,
                type_gid_rangemap,
                id_lookup,
                features,
                feat_type,
                edge_data,
            )

        # test the results
        if feat_type == "node_features":
            _validate_shuffled_nfeats(
                rank,
                world_size,
                num_chunks,
                my_feats,
                my_global_ids,
                node_feat_dim,
                node_feat_dtype,
            )
        else:
            _validate_shuffled_efeats(
                rank,
                world_size,
                num_chunks,
                num_parts,
                my_feats,
                my_global_ids,
                edge_feat_dim,
                edge_feat_dtype,
                edge_data,
            )

    except Exception as arg:
        sh_dict[f"RANK-{rank}"] = inst


def _single_machine_run(
    schema_map,
    data_dir,
    partitions_dir,
    num_chunks,
    num_parts,
    world_size,
    feat_mesg_size,
    node_feat_dim,
    node_feat_dtype,
    edge_feat_dim,
    edge_feat_dtype,
    feat_type,
):
    """Function used to spawn requested no. of process for unit testing
    Each of the spawned processes stores errors, if any during its own
    processing, in a shared dictionary. Assertion on this dictionary ensures
    success or failure of these unit test cases

    Parameters:
    -----------
    schema_map : dict
        describing the metadata of the unit test graph
    data_dir : str
        directory where the dataset is stored
    partitions_dir : str
        directory where the node-id to partition-id mappings are located
    num_chunks : int
        no. of files/chunks to generate for edges and node/edge features
    num_parts : int
        no. of graph partitions
    world_size : int
        no. of participating processes
    feat_mesg_size : int
        maximum size of the outgoing message used by ``exchange_features``
        function when shuffling node/edge features
    node_feat_dim : int
        no. of dimensions for node features. For instance, in a 2x2 matrix
        of size x rows and y columns `y` refers to feature dimensions
    node_feat_dtype : numpy dtype
        numpy dtype used when generating node features
    edge_feat_dim : int
        no. of dimensions for edge features (similar to node_feat_dim above)
    edge_feat_dtype : int
        numpy dtype used when generating edge features
    feat_type : string
        value is either `node_features` or `edge_features`
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
                rank,
                world_size,
                num_parts,
                num_chunks,
                schema_map,
                partitions_dir,
                data_dir,
                port_num,
                feat_mesg_size,
                sh_dict,
                node_feat_dim,
                node_feat_dtype,
                edge_feat_dim,
                edge_feat_dtype,
                feat_type,
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


def _prepare_test_data(
    num_chunks,
    dataset_dir,
    node_feat_dim,
    node_feat_dtype,
    edge_feat_dim,
    edge_feat_dtype,
):
    """Function to generate unit test graph. This graph has one node-type
    and two edge-types. One of the edge-types and the only node-type has
    features which are used for testing

    Parameters:
    -----------
    num_chunks : int
        no. of files/chunks to generate for edges and node/edge features
    dataset_dir : str
        directory where all the files for the test graph will be located
    node_feat_dtype : numpy dtype
        numpy dtype used when generating node features
    edge_feat_dim : int
        no. of dimensions for edge features (similar to node_feat_dim above)
    edge_feat_dtype : int
        numpy dtype used when generating edge features

    Returns:
    --------
    dictionary :
        schema dictionary describing the metadata for the unit test graph
    """
    schema = {}
    schema["num_nodes_per_type"] = [10 * num_chunks]
    schema["num_edges_per_type"] = [10 * num_chunks, 10 * num_chunks]

    schema["edge_type"] = ["n1:e1:n1", "n1:rev-e1:n1"]
    schema["node_type"] = ["n1"]

    edges = {}
    edges["n1:e1:n1"] = {}
    edges["n1:rev-e1:n1"] = {}

    edges["n1:e1:n1"]["format"] = {}
    edges["n1:rev-e1:n1"]["format"] = {}

    edges["n1:e1:n1"]["format"]["name"] = "csv"
    edges["n1:rev-e1:n1"]["format"]["name"] = "csv"

    edges["n1:e1:n1"]["format"]["delimiter"] = " "
    edges["n1:rev-e1:n1"]["format"]["delimiter"] = " "

    os.makedirs(dataset_dir, exist_ok=True)
    edges_dir = os.path.join(dataset_dir, "edges")
    os.makedirs(edges_dir, exist_ok=True)
    edges_feat_dir = os.path.join(dataset_dir, "edge_data")
    os.makedirs(edges_feat_dir, exist_ok=True)
    nodes_feat_dir = os.path.join(dataset_dir, "node_data")
    os.makedirs(nodes_feat_dir, exist_ok=True)

    fmt_meta = {"name": "csv"}
    fmt_meta["delimiter"] = " "
    edge_files = []
    rev_edge_files = []

    for idx in range(num_chunks):
        file_path = os.path.join(
            edges_dir, f"test_file_{idx}.{fmt_meta['name']}"
        )
        array_parser = array_readwriter.get_array_parser(**fmt_meta)

        src = np.arange(10)
        dst = np.arange(10)
        edge_data = np.column_stack((src, dst)) + 10 * idx

        array_parser.write(file_path, edge_data)
        edge_files.append(file_path)

        # create rev-edges here.
        rev_edge_data = edge_data
        temp = rev_edge_data[:, 0]
        rev_edge_data[:, 0] = rev_edge_data[:, 1]
        rev_edge_data[:, 1] = temp
        file_path = os.path.join(
            edges_dir, f"test_rev_file_{idx}.{fmt_meta['name']}"
        )
        array_parser.write(file_path, rev_edge_data)
        rev_edge_files.append(file_path)

    edges["n1:e1:n1"]["data"] = edge_files
    edges["n1:rev-e1:n1"]["data"] = rev_edge_files
    schema["edges"] = edges

    # create edge features.
    edge_data = {}
    edge_data["n1:e1:n1"] = {}
    edge_data["n1:e1:n1"]["edge_feat_1"] = {}

    edge_data["n1:e1:n1"]["edge_feat_1"]["format"] = {}
    edge_data["n1:e1:n1"]["edge_feat_1"]["format"]["name"] = "numpy"
    edge_data["n1:e1:n1"]["edge_feat_1"]["format"]["delimiter"] = "*"
    edge_feat_files = []

    edges_data_dir = os.path.join(dataset_dir, "edge_data")
    os.makedirs(edges_data_dir, exist_ok=True)
    for idx in range(num_chunks):
        data = np.arange(edge_feat_dim).astype(edge_feat_dtype)
        for t in range(1, 10):
            arr = (
                np.arange(t, t + edge_feat_dim).astype(edge_feat_dtype)
                + idx * 10
            )
            data = np.vstack((data, arr))
        data = data.astype(edge_feat_dtype)
        file_path = os.path.join(edges_data_dir, f"test_feat_file_{idx}.npy")
        np.save(file_path, data)
        edge_feat_files.append(file_path)

    edge_data["n1:e1:n1"]["edge_feat_1"]["data"] = edge_feat_files
    schema["edge_data"] = edge_data

    node_data = {}
    node_data["n1"] = {}
    node_data["n1"]["node_feat_1"] = {}
    node_data["n1"]["node_feat_1"]["format"] = edge_data["n1:e1:n1"][
        "edge_feat_1"
    ]["format"]

    node_feat_files = []
    nodes_data_dir = os.path.join(dataset_dir, "node_data")
    os.makedirs(nodes_data_dir, exist_ok=True)
    for idx in range(num_chunks):
        data = np.arange(node_feat_dim).astype(node_feat_dtype)
        for t in range(1, 10):
            arr = (
                np.arange(t, t + node_feat_dim).astype(node_feat_dtype)
                + idx * 10
            )
            data = np.vstack((data, arr))
        data = data.astype(node_feat_dtype)
        file_path = os.path.join(nodes_data_dir, f"test_nfeat_file_{idx}.npy")
        np.save(file_path, data)
        node_feat_files.append(file_path)

    node_data["n1"]["node_feat_1"]["data"] = node_feat_files
    schema["node_data"] = node_data

    return schema


def _prepare_data_lookupservice(partitions_dir, num_chunks, num_parts):
    """Function to generate node-id to partition-ig mappings

    Parameters:
    ----------
    partitions_dir : str
        path indicating the directory where these mappings are located
    num_chunks : int
        no. of files/chunks to generate for edges and node/edge features
    num_parts : int
        no. of graph partitions
    """
    os.makedirs(partitions_dir, exist_ok=True)
    part_file = os.path.join(partitions_dir, "n1.txt")

    num_nodes = 10 * num_chunks
    num_repeats = np.ceil(num_nodes / num_parts).astype(np.int64)

    # n1.txt
    part_ids = np.tile(np.arange(num_parts), num_repeats)[:num_nodes]
    np.savetxt(part_file, part_ids, "%d")


@pytest.mark.parametrize(
    "num_chunks, num_parts, world_size", [[1, 1, 1], [4, 2, 2]]
)
@pytest.mark.parametrize("feat_mesg_size", [0, 10])
@pytest.mark.parametrize(
    "node_feat_dim, edge_feat_dim", [[256, 256], [1024 * 4, 1024 * 4]]
)
@pytest.mark.parametrize(
    "node_feat_dtype, edge_feat_dtype",
    [[np.int32, np.int32], [np.int64, np.int64]],
)
@pytest.mark.parametrize("feat_type", ["node_features", "edge_features"])
def test_exchange_node_feats(
    num_chunks,
    num_parts,
    world_size,
    feat_mesg_size,
    node_feat_dim,
    node_feat_dtype,
    edge_feat_dim,
    edge_feat_dtype,
    feat_type,
):
    """Unit test cases to test ``exchange_features`` function. Now this function
    splits large messages into multiple smaller messages to avoid OOM issues that
    may arise when shuffling large feature data.

    Parameters:
    -----------
    num_chunks : int
        no. of files/chunks to generate for edges and node/edge features
    num_parts : int
        no. of graph partitions
    world_size : int
        no. of participating processes
    feat_mesg_size : int
        maximum size of the outgoing message used by ``exchange_features``
        function when shuffling node/edge features
    node_feat_dim : int
        no. of dimensions for node features. For instance, in a 2x2 matrix
        of size x rows and y columns `y` refers to feature dimensions
    node_feat_dtype : numpy dtype
        numpy dtype used when generating node features
    edge_feat_dim : int
        no. of dimensions for edge features (similar to node_feat_dim above)
    edge_feat_dtype : int
        numpy dtype used when generating edge features
    feat_type : string
        value is either `node_features` or `edge_features`
    """
    with tempfile.TemporaryDirectory() as root_dir:
        # Prepare the test input data
        data_dir = os.path.join(root_dir, "dataset")
        os.makedirs(data_dir, exist_ok=True)
        schema_map = _prepare_test_data(
            num_chunks,
            data_dir,
            node_feat_dim,
            node_feat_dtype,
            edge_feat_dim,
            edge_feat_dtype,
        )

        # Create the node-id to partition id mappings
        partitions_dir = os.path.join(root_dir, "partitions_dir")
        os.makedirs(partitions_dir, exist_ok=True)
        _prepare_data_lookupservice(partitions_dir, num_chunks, num_parts)

        # Fire the unit test case
        _single_machine_run(
            schema_map,
            data_dir,
            partitions_dir,
            num_chunks,
            num_parts,
            world_size,
            feat_mesg_size,
            node_feat_dim,
            node_feat_dtype,
            edge_feat_dim,
            edge_feat_dtype,
            feat_type,
        )
