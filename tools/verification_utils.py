import json
import os

import constants

import dgl

import numpy as np
import pyarrow
import pyarrow.parquet as pq
import pytest
import torch
from dgl.data.utils import load_tensors
from dgl.distributed.partition import (
    _etype_str_to_tuple,
    _etype_tuple_to_str,
    _get_inner_edge_mask,
    _get_inner_node_mask,
    RESERVED_FIELD_DTYPE,
)
from distpartitioning.utils import get_idranges


def read_file(fname, ftype):
    """Read a file from disk
    Parameters:
    -----------
    fname : string
        specifying the absolute path to the file to read
    ftype : string
        supported formats are `numpy`, `parquet', `csv`

    Returns:
    --------
    numpy ndarray :
        file contents are returned as numpy array
    """
    reader_fmt_meta = {"name": ftype}
    array_readwriter.get_array_parser(**reader_fmt_meta).read(fname)

    return data


def verify_partition_data_types(part_g):
    """Validate the dtypes in the partitioned graphs are valid

    Parameters:
    -----------
    part_g : DGL Graph object
        created for the partitioned graphs
    """
    for k, dtype in RESERVED_FIELD_DTYPE.items():
        if k in part_g.ndata:
            assert part_g.ndata[k].dtype == dtype
        if k in part_g.edata:
            assert part_g.edata[k].dtype == dtype


def verify_partition_formats(part_g, formats):
    """Validate the partitioned graphs with supported formats

    Parameters:
    -----------
    part_g : DGL Graph object
        created for the partitioned graphs
    formats : string
        formats(csc, coo, csr) supported formats and multiple
        values can be seperated by comma
    """
    # Verify saved graph formats
    if formats is None:
        assert "coo" in part_g.formats()["created"]
    else:
        formats = formats.split(",")
        for format in formats:
            assert format in part_g.formats()["created"]


def verify_graph_feats(
    g, gpb, part, node_feats, edge_feats, orig_nids, orig_eids
):
    """Verify the node/edge features of the partitioned graph with
    the original graph

    Parameters:
    -----------
    g : DGL Graph Object
        of the original graph
    gpb : global partition book
        created for the partitioned graph object
    node_feats : dictionary
        with key, value pairs as node-types and features as numpy arrays
    edge_feats : dictionary
        with key, value pairs as edge-types and features as numpy arrays
    orig_nids : dictionary
        with key, value pairs as node-types and (global) nids from the
        original graph
    orig_eids : dictionary
        with key, value pairs as edge-types and (global) eids from the
        original graph
    """
    for ntype in g.ntypes:
        ntype_id = g.get_ntype_id(ntype)
        inner_node_mask = _get_inner_node_mask(part, ntype_id)
        inner_nids = part.ndata[dgl.NID][inner_node_mask]
        ntype_ids, inner_type_nids = gpb.map_to_per_ntype(inner_nids)
        partid = gpb.nid2partid(inner_type_nids, ntype)
        assert np.all(ntype_ids.numpy() == ntype_id)
        assert np.all(partid.numpy() == gpb.partid)

        orig_id = orig_nids[ntype][inner_type_nids]
        local_nids = gpb.nid2localnid(inner_type_nids, gpb.partid, ntype)

        for name in g.nodes[ntype].data:
            if name in [dgl.NID, "inner_node"]:
                continue
            true_feats = g.nodes[ntype].data[name][orig_id]
            ndata = node_feats[ntype + "/" + name][local_nids]
            assert np.array_equal(ndata.numpy(), true_feats.numpy())

    for etype in g.canonical_etypes:
        etype_id = g.get_etype_id(etype)
        inner_edge_mask = _get_inner_edge_mask(part, etype_id)
        inner_eids = part.edata[dgl.EID][inner_edge_mask]
        etype_ids, inner_type_eids = gpb.map_to_per_etype(inner_eids)
        partid = gpb.eid2partid(inner_type_eids, etype)
        assert np.all(etype_ids.numpy() == etype_id)
        assert np.all(partid.numpy() == gpb.partid)

        orig_id = orig_eids[_etype_tuple_to_str(etype)][inner_type_eids]
        local_eids = gpb.eid2localeid(inner_type_eids, gpb.partid, etype)

        for name in g.edges[etype].data:
            if name in [dgl.EID, "inner_edge"]:
                continue
            true_feats = g.edges[etype].data[name][orig_id]
            edata = edge_feats[_etype_tuple_to_str(etype) + "/" + name][
                local_eids
            ]
            assert np.array_equal(edata.numpy(), true_feats.numpy())


def verify_metadata_counts(part_schema, part_g, graph_schema, g, partid):
    """Verify the partitioned graph objects with the metadata

    Parameters:
    -----------
    part_schema : json object
        which is created by reading the metadata.json file for the
        partitioned graph
    part_g : DGL graph object
        of a graph partition
    graph_schema : json object
        which is created by reading the metadata.json file for the
        original graph
    g : DGL Graph object
        created by reading the original graph from the disk.
    partid : integer
        specifying the partition id of the graph object, part_g
    """
    for ntype in part_schema[constants.STR_NTYPES]:
        ntype_data = part_schema[constants.STR_NODE_MAP][ntype]
        meta_ntype_count = ntype_data[partid][1] - ntype_data[partid][0]
        inner_node_mask = _get_inner_node_mask(part_g, g.get_ntype_id(ntype))
        graph_ntype_count = len(part_g.ndata[dgl.NID][inner_node_mask])
        assert (
            meta_ntype_count == graph_ntype_count
        ), f"Metadata ntypecount = {meta_ntype_count} and graph_ntype_count = {graph_ntype_count}"

    for etype in part_schema[constants.STR_ETYPES]:
        etype_data = part_schema[constants.STR_EDGE_MAP][etype]
        meta_etype_count = etype_data[partid][1] - etype_data[partid][0]
        mask = _get_inner_edge_mask(
            part_g, g.get_etype_id(_etype_str_to_tuple(etype))
        )
        graph_etype_count = len(part_g.edata[dgl.EID][mask])
        assert (
            meta_etype_count == graph_etype_count
        ), f"Metadata etypecount = {meta_etype_count} does not match part graph etypecount = {graph_etype_count}"


def get_node_partids(partitions_dir, graph_schema):
    """load the node partition ids from the disk

    Parameters:
    ----------
    partitions_dir : string
        directory path where metis/random partitions are located
    graph_schema : json object
        which is created by reading the metadata.json file for the
        original graph

    Returns:
    --------
    dictionary :
        where keys are node-types and value is a list of partition-ids for all the
        nodes of that particular node-type.
    """
    assert os.path.isdir(
        partitions_dir
    ), f"Please provide a valid directory to read nodes to partition-id mappings."
    _, gid_dict = get_idranges(
        graph_schema[constants.STR_NODE_TYPE],
        dict(
            zip(
                graph_schema[constants.STR_NODE_TYPE],
                graph_schema[constants.STR_NODE_TYPE_COUNTS],
            )
        ),
    )
    node_partids = {}
    for ntype_id, ntype in enumerate(graph_schema[constants.STR_NODE_TYPE]):
        node_partids[ntype] = read_file(
            os.path.join(partitions_dir, f"{ntype}.txt"), constants.STR_CSV
        )
        assert (
            len(node_partids[ntype])
            == graph_schema[constants.STR_NODE_TYPE_COUNTS][ntype_id]
        ), f"Node count for {ntype} = {len(node_partids[ntype])} in the partitions_dir while it should be {graph_schema[constants.STR_NTYPE_COUNTS][ntype_id]} (from graph schema)."

    return node_partids


def verify_node_partitionids(
    node_partids, part_g, g, gpb, graph_schema, orig_nids, partition_id
):
    """Verify partitioned graph objects node counts with the original graph

    Parameters:
    -----------
    params : argparser object
        to access command line arguments for this python script
    part_data : list of tuples
        partitioned graph objects read from the disk
    g : DGL Graph object
        created by reading the original graph from disk
    graph_schema : json object
        created by reading the metadata.json file for the original graph
    orig_nids : dictionary
        which contains the origial(global) node-ids
    partition_id : integer
        partition id of the partitioned graph, part_g
    """
    # read part graphs and verify the counts
    # inner node masks, should give the node counts in each part-g and get the corresponding orig-ids to map to the original graph node-ids
    for ntype_id, ntype in enumerate(graph_schema[constants.STR_NODE_TYPE]):
        mask = _get_inner_node_mask(part_g, g.get_ntype_id(ntype))

        # map these to orig-nids.
        inner_nids = part_g.ndata[dgl.NID][mask]
        ntype_ids, inner_type_nids = gpb.map_to_per_ntype(inner_nids)
        partid = gpb.nid2partid(inner_type_nids, ntype)

        assert np.all(ntype_ids.numpy() == ntype_id)
        assert np.all(partid.numpy() == gpb.partid)

        idxes = orig_nids[ntype][inner_type_nids]
        assert np.all(idxes >= 0)

        # get the partition-ids for these nodes.
        assert np.all(
            node_partids[ntype][idxes] == partition_id
        ), f"All the nodes in the partition = {partid} does not their nodeid to partition-id maps are defined by the partitioning algorithm. Node-type = {ntype}"


def read_orig_ids(out_dir, fname, num_parts):
    """Read original id files for the partitioned graph objects

    Parameters:
    -----------
    out_dir : string
        specifying the directory where the files are located
    fname : string
        file name to read from
    num_parts : integer
        no. of partitions

    Returns:
    --------
    dictionary :
        where keys are node/edge types and values are original node
        or edge ids from the original graph
    """
    orig_ids = {}
    for i in range(num_parts):
        ids_path = os.path.join(out_dir, f"part{i}", fname)
        part_ids = load_tensors(ids_path)
        for type, data in part_ids.items():
            if type not in orig_ids:
                orig_ids[type] = data.numpy()
            else:
                orig_ids[type] = np.concatenate((orig_ids[type], data))
    return orig_ids
