import json
import os
import logging
from typing import Dict, Any

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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PartitionVerificationError(Exception):
    """Custom exception for partition verification errors."""
    pass

def read_file(fname: str, ftype: str) -> np.ndarray:
    """Read a file from disk"""
    try:
        reader_fmt_meta = {"name": ftype}
        data = array_readwriter.get_array_parser(**reader_fmt_meta).read(fname)
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {fname}")
        raise
    except Exception as e:
        logging.error(f"Error reading file {fname}: {str(e)}")
        raise PartitionVerificationError(f"Failed to read file {fname}") from e

def verify_partition_data_types(part_g: dgl.DGLGraph):
    """Validate the dtypes in the partitioned graphs are valid"""
    try:
        for k, dtype in RESERVED_FIELD_DTYPE.items():
            if k in part_g.ndata:
                assert part_g.ndata[k].dtype == dtype, f"Incorrect dtype for ndata[{k}]"
            if k in part_g.edata:
                assert part_g.edata[k].dtype == dtype, f"Incorrect dtype for edata[{k}]"
    except AssertionError as e:
        logging.error(f"Data type verification failed: {str(e)}")
        raise PartitionVerificationError("Partition data type verification failed") from e

def verify_partition_formats(part_g: dgl.DGLGraph, formats: str):
    """Validate the partitioned graphs with supported formats"""
    try:
        if formats is None:
            assert "coo" in part_g.formats()["created"], "COO format not found"
        else:
            formats = formats.split(",")
            for format in formats:
                assert format in part_g.formats()["created"], f"{format} format not found"
    except AssertionError as e:
        logging.error(f"Format verification failed: {str(e)}")
        raise PartitionVerificationError("Partition format verification failed") from e

def verify_graph_feats(
    g: dgl.DGLGraph,
    gpb: Any,
    part: Any,
    node_feats: Dict[str, np.ndarray],
    edge_feats: Dict[str, np.ndarray],
    orig_nids: Dict[str, np.ndarray],
    orig_eids: Dict[str, np.ndarray]
):
    """Verify the node/edge features of the partitioned graph with the original graph"""
    try:
        for ntype in g.ntypes:
            ntype_id = g.get_ntype_id(ntype)
            inner_node_mask = _get_inner_node_mask(part, ntype_id)
            inner_nids = part.ndata[dgl.NID][inner_node_mask]
            ntype_ids, inner_type_nids = gpb.map_to_per_ntype(inner_nids)
            partid = gpb.nid2partid(inner_type_nids, ntype)
            assert np.all(ntype_ids.numpy() == ntype_id), f"Incorrect ntype_ids for {ntype}"
            assert np.all(partid.numpy() == gpb.partid), f"Incorrect partid for {ntype}"

            orig_id = orig_nids[ntype][inner_type_nids]
            local_nids = gpb.nid2localnid(inner_type_nids, gpb.partid, ntype)

            for name in g.nodes[ntype].data:
                if name in [dgl.NID, "inner_node"]:
                    continue
                true_feats = g.nodes[ntype].data[name][orig_id]
                ndata = node_feats[ntype + "/" + name][local_nids]
                assert np.array_equal(ndata.numpy(), true_feats.numpy()), f"Mismatch in node features for {ntype}/{name}"

        for etype in g.canonical_etypes:
            etype_id = g.get_etype_id(etype)
            inner_edge_mask = _get_inner_edge_mask(part, etype_id)
            inner_eids = part.edata[dgl.EID][inner_edge_mask]
            etype_ids, inner_type_eids = gpb.map_to_per_etype(inner_eids)
            partid = gpb.eid2partid(inner_type_eids, etype)
            assert np.all(etype_ids.numpy() == etype_id), f"Incorrect etype_ids for {etype}"
            assert np.all(partid.numpy() == gpb.partid), f"Incorrect partid for {etype}"

            orig_id = orig_eids[_etype_tuple_to_str(etype)][inner_type_eids]
            local_eids = gpb.eid2localeid(inner_type_eids, gpb.partid, etype)

            for name in g.edges[etype].data:
                if name in [dgl.EID, "inner_edge"]:
                    continue
                true_feats = g.edges[etype].data[name][orig_id]
                edata = edge_feats[_etype_tuple_to_str(etype) + "/" + name][local_eids]
                assert np.array_equal(edata.numpy(), true_feats.numpy()), f"Mismatch in edge features for {etype}/{name}"

    except AssertionError as e:
        logging.error(f"Graph feature verification failed: {str(e)}")
        raise PartitionVerificationError("Graph feature verification failed") from e
    except KeyError as e:
        logging.error(f"Missing key in graph data: {str(e)}")
        raise PartitionVerificationError("Graph feature verification failed due to missing data") from e

def verify_metadata_counts(part_schema: Dict[str, Any], part_g: dgl.DGLGraph, graph_schema: Dict[str, Any], g: dgl.DGLGraph, partid: int):
    """Verify the partitioned graph objects with the metadata"""
    try:
        for ntype in part_schema[constants.STR_NTYPES]:
            ntype_data = part_schema[constants.STR_NODE_MAP][ntype]
            meta_ntype_count = ntype_data[partid][1] - ntype_data[partid][0]
            inner_node_mask = _get_inner_node_mask(part_g, g.get_ntype_id(ntype))
            graph_ntype_count = len(part_g.ndata[dgl.NID][inner_node_mask])
            assert meta_ntype_count == graph_ntype_count, f"Metadata ntype count ({meta_ntype_count}) does not match graph ntype count ({graph_ntype_count}) for {ntype}"

        for etype in part_schema[constants.STR_ETYPES]:
            etype_data = part_schema[constants.STR_EDGE_MAP][etype]
            meta_etype_count = etype_data[partid][1] - etype_data[partid][0]
            mask = _get_inner_edge_mask(part_g, g.get_etype_id(_etype_str_to_tuple(etype)))
            graph_etype_count = len(part_g.edata[dgl.EID][mask])
            assert meta_etype_count == graph_etype_count, f"Metadata etype count ({meta_etype_count}) does not match graph etype count ({graph_etype_count}) for {etype}"

    except AssertionError as e:
        logging.error(f"Metadata count verification failed: {str(e)}")
        raise PartitionVerificationError("Metadata count verification failed") from e
    except KeyError as e:
        logging.error(f"Missing key in schema: {str(e)}")
        raise PartitionVerificationError("Metadata count verification failed due to missing schema data") from e

def get_node_partids(partitions_dir: str, graph_schema: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """load the node partition ids from the disk"""
    try:
        assert os.path.isdir(partitions_dir), f"Invalid directory: {partitions_dir}"
        
        _, gid_dict = get_idranges(
            graph_schema[constants.STR_NODE_TYPE],
            dict(zip(graph_schema[constants.STR_NODE_TYPE], graph_schema[constants.STR_NODE_TYPE_COUNTS])),
        )
        
        node_partids = {}
        for ntype_id, ntype in enumerate(graph_schema[constants.STR_NODE_TYPE]):
            node_partids[ntype] = read_file(os.path.join(partitions_dir, f"{ntype}.txt"), constants.STR_CSV)
            assert len(node_partids[ntype]) == graph_schema[constants.STR_NODE_TYPE_COUNTS][ntype_id], \
                f"Node count mismatch for {ntype}: {len(node_partids[ntype])} vs {graph_schema[constants.STR_NTYPE_COUNTS][ntype_id]}"
        
        return node_partids
    
    except AssertionError as e:
        logging.error(f"Node partition ID loading failed: {str(e)}")
        raise PartitionVerificationError("Failed to load node partition IDs") from e
    except Exception as e:
        logging.error(f"Unexpected error in get_node_partids: {str(e)}")
        raise PartitionVerificationError("Failed to load node partition IDs") from e

def verify_node_partitionids(
    node_partids: Dict[str, np.ndarray],
    part_g: dgl.DGLGraph,
    g: dgl.DGLGraph,
    gpb: Any,
    graph_schema: Dict[str, Any],
    orig_nids: Dict[str, np.ndarray],
    partition_id: int
):
    """Verify partitioned graph objects node counts with the original graph"""
    try:
        for ntype_id, ntype in enumerate(graph_schema[constants.STR_NODE_TYPE]):
            mask = _get_inner_node_mask(part_g, g.get_ntype_id(ntype))
            inner_nids = part_g.ndata[dgl.NID][mask]
            ntype_ids, inner_type_nids = gpb.map_to_per_ntype(inner_nids)
            partid = gpb.nid2partid(inner_type_nids, ntype)

            assert np.all(ntype_ids.numpy() == ntype_id), f"Incorrect ntype_ids for {ntype}"
            assert np.all(partid.numpy() == gpb.partid), f"Incorrect partid for {ntype}"

            idxes = orig_nids[ntype][inner_type_nids]
            assert np.all(idxes >= 0), f"Invalid original node IDs for {ntype}"

            assert np.all(node_partids[ntype][idxes] == partition_id), \
                f"Partition ID mismatch for nodes in {ntype}"

    except AssertionError as e:
        logging.error(f"Node partition ID verification failed: {str(e)}")
        raise PartitionVerificationError("Node partition ID verification failed") from e
    except KeyError as e:
        logging.error(f"Missing key in node data: {str(e)}")
        raise PartitionVerificationError("Node partition ID verification failed due to missing data") from e

def read_orig_ids(out_dir: str, fname: str, num_parts: int) -> Dict[str, np.ndarray]:
    """Read original id files for the partitioned graph objects"""
    try:
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
    except FileNotFoundError as e:
        logging.error(f"Original ID file not found: {str(e)}")
        raise PartitionVerificationError(f"Failed to read original IDs from {fname}") from e
    except Exception as e:
        logging.error(f"Error reading original IDs: {str(e)}")
        raise PartitionVerificationError(f"Failed to read original IDs from {fname}") from e