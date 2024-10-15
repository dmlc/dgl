"""Functions for partitions. """

import concurrent
import concurrent.futures
import copy
import json
import logging
import multiprocessing as mp
import os
import time
from functools import partial

import numpy as np

import torch

from .. import backend as F, graphbolt as gb
from ..base import dgl_warning, DGLError, EID, ETYPE, NID, NTYPE
from ..convert import heterograph, to_homogeneous
from ..data.utils import load_graphs, load_tensors, save_graphs, save_tensors
from ..partition import (
    get_peak_mem,
    metis_partition_assignment,
    partition_graph_with_halo,
)
from ..random import choice as random_choice
from ..transforms import sort_csc_by_tag, sort_csr_by_tag
from .constants import DEFAULT_ETYPE, DEFAULT_NTYPE, DGL2GB_EID, GB_DST_ID
from .graph_partition_book import (
    _etype_str_to_tuple,
    _etype_tuple_to_str,
    RangePartitionBook,
)


RESERVED_FIELD_DTYPE = {
    "inner_node": (
        F.uint8
    ),  # A flag indicates whether the node is inside a partition.
    "inner_edge": (
        F.uint8
    ),  # A flag indicates whether the edge is inside a partition.
    NID: F.int64,
    EID: F.int64,
    NTYPE: F.int16,
    # `sort_csr_by_tag` and `sort_csc_by_tag` works on int32/64 only.
    ETYPE: F.int32,
}


def _format_part_metadata(part_metadata, formatter):
    """Format etypes with specified formatter."""
    for key in ["edge_map", "etypes"]:
        if key not in part_metadata:
            continue
        orig_data = part_metadata[key]
        if not isinstance(orig_data, dict):
            continue
        new_data = {}
        for etype, data in orig_data.items():
            etype = formatter(etype)
            new_data[etype] = data
        part_metadata[key] = new_data
    return part_metadata


def _load_part_config(part_config):
    """Load part config and format."""
    try:
        with open(part_config) as f:
            part_metadata = _format_part_metadata(
                json.load(f), _etype_str_to_tuple
            )
    except AssertionError as e:
        raise DGLError(
            f"Failed to load partition config due to {e}. "
            "Probably caused by outdated config. If so, please refer to "
            "https://github.com/dmlc/dgl/tree/master/tools#change-edge-"
            "type-to-canonical-edge-type-for-partition-configuration-json"
        )
    return part_metadata


def _dump_part_config(part_config, part_metadata):
    """Format and dump part config."""
    part_metadata = _format_part_metadata(part_metadata, _etype_tuple_to_str)
    with open(part_config, "w") as outfile:
        json.dump(part_metadata, outfile, sort_keys=False, indent=4)


def process_partitions(g, formats=None, sort_etypes=False):
    """Preprocess partitions before saving:
    1. format data types.
    2. sort csc/csr by tag.
    """
    for k, dtype in RESERVED_FIELD_DTYPE.items():
        if k in g.ndata:
            g.ndata[k] = F.astype(g.ndata[k], dtype)
        if k in g.edata:
            g.edata[k] = F.astype(g.edata[k], dtype)

    if (sort_etypes) and (formats is not None):
        if "csr" in formats:
            g = sort_csr_by_tag(g, tag=g.edata[ETYPE], tag_type="edge")
        if "csc" in formats:
            g = sort_csc_by_tag(g, tag=g.edata[ETYPE], tag_type="edge")
    return g


def _save_dgl_graphs(filename, g_list, formats=None):
    save_graphs(filename, g_list, formats=formats)


def _get_inner_node_mask(graph, ntype_id, gpb=None):
    ndata = (
        graph.node_attributes
        if isinstance(graph, gb.FusedCSCSamplingGraph)
        else graph.ndata
    )
    assert "inner_node" in ndata, "'inner_node' is not in nodes' data"
    if NTYPE in ndata or gpb is not None:
        ntype = (
            gpb.map_to_per_ntype(ndata[NID])[0]
            if gpb is not None
            else ndata[NTYPE]
        )
        dtype = F.dtype(ndata["inner_node"])
        return ndata["inner_node"] * F.astype(ntype == ntype_id, dtype) == 1
    else:
        return ndata["inner_node"] == 1


def _get_inner_edge_mask(
    graph,
    etype_id,
):
    edata = (
        graph.edge_attributes
        if isinstance(graph, gb.FusedCSCSamplingGraph)
        else graph.edata
    )
    assert "inner_edge" in edata, "'inner_edge' is not in edges' data"
    etype = (
        graph.type_per_edge
        if isinstance(graph, gb.FusedCSCSamplingGraph)
        else (graph.edata[ETYPE] if ETYPE in graph.edata else None)
    )
    if etype is not None:
        dtype = F.dtype(edata["inner_edge"])
        return edata["inner_edge"] * F.astype(etype == etype_id, dtype) == 1
    else:
        return edata["inner_edge"] == 1


def _get_part_ranges(id_ranges):
    res = {}
    for key in id_ranges:
        # Normally, each element has two values that represent the starting ID and the ending ID
        # of the ID range in a partition.
        # If not, the data is probably still in the old format, in which only the ending ID is
        # stored. We need to convert it to the format we expect.
        if not isinstance(id_ranges[key][0], list):
            start = 0
            for i, end in enumerate(id_ranges[key]):
                id_ranges[key][i] = [start, end]
                start = end
        res[key] = np.concatenate(
            [np.array(l) for l in id_ranges[key]]
        ).reshape(-1, 2)
    return res


def _verify_dgl_partition(graph, part_id, gpb, ntypes, etypes):
    """Verify the partition of a DGL graph."""
    assert (
        NID in graph.ndata
    ), "the partition graph should contain node mapping to global node ID"
    assert (
        EID in graph.edata
    ), "the partition graph should contain edge mapping to global edge ID"

    for ntype in ntypes:
        ntype_id = ntypes[ntype]
        # graph.ndata[NID] are global homogeneous node IDs.
        nids = F.boolean_mask(
            graph.ndata[NID], _get_inner_node_mask(graph, ntype_id)
        )
        partids1 = gpb.nid2partid(nids)
        _, per_type_nids = gpb.map_to_per_ntype(nids)
        partids2 = gpb.nid2partid(per_type_nids, ntype)
        assert np.all(F.asnumpy(partids1 == part_id)), (
            "Unexpected partition IDs are found in the loaded partition "
            "while querying via global homogeneous node IDs."
        )
        assert np.all(F.asnumpy(partids2 == part_id)), (
            "Unexpected partition IDs are found in the loaded partition "
            "while querying via type-wise node IDs."
        )
    for etype in etypes:
        etype_id = etypes[etype]
        # graph.edata[EID] are global homogeneous edge IDs.
        eids = F.boolean_mask(
            graph.edata[EID], _get_inner_edge_mask(graph, etype_id)
        )
        partids1 = gpb.eid2partid(eids)
        _, per_type_eids = gpb.map_to_per_etype(eids)
        partids2 = gpb.eid2partid(per_type_eids, etype)
        assert np.all(F.asnumpy(partids1 == part_id)), (
            "Unexpected partition IDs are found in the loaded partition "
            "while querying via global homogeneous edge IDs."
        )
        assert np.all(F.asnumpy(partids2 == part_id)), (
            "Unexpected partition IDs are found in the loaded partition "
            "while querying via type-wise edge IDs."
        )


def _verify_graphbolt_partition(graph, part_id, gpb, ntypes, etypes):
    """Verify the partition of a GraphBolt graph."""
    required_ndata_fields = [NID]
    required_edata_fields = [EID]
    assert all(
        field in graph.node_attributes for field in required_ndata_fields
    ), "the partition graph should contain node mapping to global node ID."
    assert all(
        field in graph.edge_attributes for field in required_edata_fields
    ), "the partition graph should contain edge mapping to global edge ID."

    num_edges = graph.total_num_edges
    local_src_ids = graph.indices
    local_dst_ids = gb.expand_indptr(
        graph.csc_indptr, dtype=local_src_ids.dtype, output_size=num_edges
    )
    global_src_ids = graph.node_attributes[NID][local_src_ids]
    global_dst_ids = graph.node_attributes[NID][local_dst_ids]

    etype_ids, type_wise_eids = gpb.map_to_per_etype(graph.edge_attributes[EID])
    if graph.type_per_edge is not None:
        assert torch.equal(etype_ids, graph.type_per_edge)
    etype_ids, etype_ids_indices = torch.sort(etype_ids)
    global_src_ids = global_src_ids[etype_ids_indices]
    global_dst_ids = global_dst_ids[etype_ids_indices]
    type_wise_eids = type_wise_eids[etype_ids_indices]

    src_ntype_ids, src_type_wise_nids = gpb.map_to_per_ntype(global_src_ids)
    dst_ntype_ids, dst_type_wise_nids = gpb.map_to_per_ntype(global_dst_ids)

    data_dict = dict()
    edge_ids = dict()
    for c_etype, etype_id in etypes.items():
        idx = etype_ids == etype_id
        src_ntype, etype, dst_ntype = c_etype
        if idx.sum() == 0:
            continue
        actual_src_ntype_ids = src_ntype_ids[idx]
        actual_dst_ntype_ids = dst_ntype_ids[idx]
        expected_src_ntype_ids = ntypes[src_ntype]
        expected_dst_ntype_ids = ntypes[dst_ntype]
        assert all(actual_src_ntype_ids == expected_src_ntype_ids), (
            f"Unexpected types of source nodes for {c_etype}. Expected: "
            f"{expected_src_ntype_ids}, but got: {actual_src_ntype_ids}."
        )
        assert all(actual_dst_ntype_ids == expected_dst_ntype_ids), (
            f"Unexpected types of destination nodes for {c_etype}. Expected: "
            f"{expected_dst_ntype_ids}, but got: {actual_dst_ntype_ids}."
        )
        data_dict[c_etype] = (src_type_wise_nids[idx], dst_type_wise_nids[idx])
        edge_ids[c_etype] = type_wise_eids[idx]

    # Make sure node/edge IDs are not out of range.
    hg = heterograph(
        data_dict, {ntype: gpb._num_nodes(ntype) for ntype in ntypes}
    )
    for etype in edge_ids:
        hg.edges[etype].data[EID] = edge_ids[etype]
    assert all(
        hg.num_edges(etype) == len(eids) for etype, eids in edge_ids.items()
    ), "The number of edges per etype in the partition graph is not correct."
    assert num_edges == hg.num_edges(), (
        f"The total number of edges in the partition graph is not correct. "
        f"Expected: {num_edges}, but got: {hg.num_edges()}."
    )
    print(f"Partition {part_id} looks good!")


def load_partition(part_config, part_id, load_feats=True, use_graphbolt=False):
    """Load data of a partition from the data path.

    A partition data includes a graph structure of the partition, a dict of node tensors,
    a dict of edge tensors and some metadata. The partition may contain the HALO nodes,
    which are the nodes replicated from other partitions. However, the dict of node tensors
    only contains the node data that belongs to the local partition. Similarly, edge tensors
    only contains the edge data that belongs to the local partition. The metadata include
    the information of the global graph (not the local partition), which includes the number
    of nodes, the number of edges as well as the node assignment of the global graph.

    The function currently loads data through the local filesystem interface.

    Parameters
    ----------
    part_config : str
        The path of the partition config file.
    part_id : int
        The partition ID.
    load_feats : bool, optional
        Whether to load node/edge feats. If False, the returned node/edge feature
        dictionaries will be empty. Default: True.
    use_graphbolt : bool, optional
        Whether to load GraphBolt partition. Default: False.

    Returns
    -------
    DGLGraph
        The graph partition structure.
    Dict[str, Tensor]
        Node features.
    Dict[(str, str, str), Tensor]
        Edge features.
    GraphPartitionBook
        The graph partition information.
    str
        The graph name
    List[str]
        The node types
    List[(str, str, str)]
        The edge types
    """
    config_path = os.path.dirname(part_config)
    relative_to_config = lambda path: os.path.join(config_path, path)

    with open(part_config) as conf_f:
        part_metadata = json.load(conf_f)
    assert (
        "part-{}".format(part_id) in part_metadata
    ), "part-{} does not exist".format(part_id)
    part_files = part_metadata["part-{}".format(part_id)]

    exist_dgl_graph = exist_graphbolt_graph = False
    if os.path.exists(os.path.join(config_path, f"part{part_id}", "graph.dgl")):
        use_graphbolt = False
        exist_dgl_graph = True
    if os.path.exists(
        os.path.join(
            config_path, f"part{part_id}", "fused_csc_sampling_graph.pt"
        )
    ):
        use_graphbolt = True
        exist_graphbolt_graph = True

    # Check if both DGL graph and GraphBolt graph exist or not exist. Make sure only one exists.
    if not exist_dgl_graph and not exist_graphbolt_graph:
        raise ValueError("The graph object doesn't exist.")
    if exist_dgl_graph and exist_graphbolt_graph:
        raise ValueError(
            "Both DGL graph and GraphBolt graph exist. Please remove one."
        )

    if use_graphbolt:
        part_graph_field = "part_graph_graphbolt"
    else:
        part_graph_field = "part_graph"
    assert (
        part_graph_field in part_files
    ), f"the partition does not contain graph structure: {part_graph_field}"
    partition_path = relative_to_config(part_files[part_graph_field])
    logging.info(
        "Start to load partition from %s which is "
        "%d bytes. It may take non-trivial "
        "time for large partition.",
        partition_path,
        os.path.getsize(partition_path),
    )
    graph = (
        torch.load(partition_path)
        if use_graphbolt
        else load_graphs(partition_path)[0][0]
    )
    logging.info("Finished loading partition from %s.", partition_path)

    gpb, graph_name, ntypes, etypes = load_partition_book(part_config, part_id)
    ntypes_list = list(ntypes.keys())
    etypes_list = list(etypes.keys())

    if "DGL_DIST_DEBUG" in os.environ:
        _verify_func = (
            _verify_graphbolt_partition
            if use_graphbolt
            else _verify_dgl_partition
        )
        _verify_func(graph, part_id, gpb, ntypes, etypes)

    node_feats = {}
    edge_feats = {}
    if load_feats:
        node_feats, edge_feats = load_partition_feats(part_config, part_id)

    return (
        graph,
        node_feats,
        edge_feats,
        gpb,
        graph_name,
        ntypes_list,
        etypes_list,
    )


def load_partition_feats(
    part_config, part_id, load_nodes=True, load_edges=True
):
    """Load node/edge feature data from a partition.

    Parameters
    ----------
    part_config : str
        The path of the partition config file.
    part_id : int
        The partition ID.
    load_nodes : bool, optional
        Whether to load node features. If ``False``, ``None`` is returned.
    load_edges : bool, optional
        Whether to load edge features. If ``False``, ``None`` is returned.

    Returns
    -------
    Dict[str, Tensor] or None
        Node features.
    Dict[str, Tensor] or None
        Edge features.
    """
    config_path = os.path.dirname(part_config)
    relative_to_config = lambda path: os.path.join(config_path, path)

    with open(part_config) as conf_f:
        part_metadata = json.load(conf_f)
    assert (
        "part-{}".format(part_id) in part_metadata
    ), "part-{} does not exist".format(part_id)
    part_files = part_metadata["part-{}".format(part_id)]
    assert (
        "node_feats" in part_files
    ), "the partition does not contain node features."
    assert (
        "edge_feats" in part_files
    ), "the partition does not contain edge feature."
    node_feats = None
    if load_nodes:
        feat_path = relative_to_config(part_files["node_feats"])
        logging.debug(
            "Start to load node data from %s which is " "%d bytes.",
            feat_path,
            os.path.getsize(feat_path),
        )
        node_feats = load_tensors(feat_path)
        logging.info("Finished loading node data.")
    edge_feats = None
    if load_edges:
        feat_path = relative_to_config(part_files["edge_feats"])
        logging.debug(
            "Start to load edge data from %s which is " "%d bytes.",
            feat_path,
            os.path.getsize(feat_path),
        )
        edge_feats = load_tensors(feat_path)
        logging.info("Finished loading edge data.")
    # In the old format, the feature name doesn't contain node/edge type.
    # For compatibility, let's add node/edge types to the feature names.
    if node_feats is not None:
        new_feats = {}
        for name in node_feats:
            feat = node_feats[name]
            if name.find("/") == -1:
                name = DEFAULT_NTYPE + "/" + name
            new_feats[name] = feat
        node_feats = new_feats
    if edge_feats is not None:
        new_feats = {}
        for name in edge_feats:
            feat = edge_feats[name]
            if name.find("/") == -1:
                name = _etype_tuple_to_str(DEFAULT_ETYPE) + "/" + name
            new_feats[name] = feat
        edge_feats = new_feats

    return node_feats, edge_feats


def load_partition_book(part_config, part_id, part_metadata=None):
    """Load a graph partition book from the partition config file.

    Parameters
    ----------
    part_config : str
        The path of the partition config file.
    part_id : int
        The partition ID.
    part_metadata : dict
        The meta data of partition.

    Returns
    -------
    GraphPartitionBook
        The global partition information.
    str
        The graph name
    dict
        The node types
    dict
        The edge types
    """
    if part_metadata is None:
        part_metadata = _load_part_config(part_config)
    assert "num_parts" in part_metadata, "num_parts does not exist."
    assert (
        part_metadata["num_parts"] > part_id
    ), "part {} is out of range (#parts: {})".format(
        part_id, part_metadata["num_parts"]
    )
    num_parts = part_metadata["num_parts"]
    assert (
        "num_nodes" in part_metadata
    ), "cannot get the number of nodes of the global graph."
    assert (
        "num_edges" in part_metadata
    ), "cannot get the number of edges of the global graph."
    assert "node_map" in part_metadata, "cannot get the node map."
    assert "edge_map" in part_metadata, "cannot get the edge map."
    assert "graph_name" in part_metadata, "cannot get the graph name"

    # If this is a range partitioning, node_map actually stores a list, whose elements
    # indicate the boundary of range partitioning. Otherwise, node_map stores a filename
    # that contains node map in a NumPy array.
    node_map = part_metadata["node_map"]
    edge_map = part_metadata["edge_map"]
    if isinstance(node_map, dict):
        for key in node_map:
            is_range_part = isinstance(node_map[key], list)
            break
    elif isinstance(node_map, list):
        is_range_part = True
        node_map = {DEFAULT_NTYPE: node_map}
    else:
        is_range_part = False
    if isinstance(edge_map, list):
        edge_map = {DEFAULT_ETYPE: edge_map}

    ntypes = {DEFAULT_NTYPE: 0}
    etypes = {DEFAULT_ETYPE: 0}
    if "ntypes" in part_metadata:
        ntypes = part_metadata["ntypes"]
    if "etypes" in part_metadata:
        etypes = part_metadata["etypes"]

    if isinstance(node_map, dict):
        for key in node_map:
            assert key in ntypes, "The node type {} is invalid".format(key)
    if isinstance(edge_map, dict):
        for key in edge_map:
            assert key in etypes, "The edge type {} is invalid".format(key)

    if not is_range_part:
        raise TypeError("Only RangePartitionBook is supported currently.")

    node_map = _get_part_ranges(node_map)
    edge_map = _get_part_ranges(edge_map)

    # Format dtype of node/edge map if dtype is specified.
    def _format_node_edge_map(part_metadata, map_type, data):
        key = f"{map_type}_map_dtype"
        if key not in part_metadata:
            return data
        dtype = part_metadata[key]
        assert dtype in ["int32", "int64"], (
            f"The {map_type} map dtype should be either int32 or int64, "
            f"but got {dtype}."
        )
        for key in data:
            data[key] = data[key].astype(dtype)
        return data

    node_map = _format_node_edge_map(part_metadata, "node", node_map)
    edge_map = _format_node_edge_map(part_metadata, "edge", edge_map)

    # Sort the node/edge maps by the node/edge type ID.
    node_map = dict(sorted(node_map.items(), key=lambda x: ntypes[x[0]]))
    edge_map = dict(sorted(edge_map.items(), key=lambda x: etypes[x[0]]))

    def _assert_is_sorted(id_map):
        id_ranges = np.array(list(id_map.values()))
        ids = []
        for i in range(num_parts):
            ids.append(id_ranges[:, i, :])
        ids = np.array(ids).flatten()
        assert np.all(
            ids[:-1] <= ids[1:]
        ), f"The node/edge map is not sorted: {ids}"

    _assert_is_sorted(node_map)
    _assert_is_sorted(edge_map)

    return (
        RangePartitionBook(
            part_id, num_parts, node_map, edge_map, ntypes, etypes
        ),
        part_metadata["graph_name"],
        ntypes,
        etypes,
    )


def _get_orig_ids(g, sim_g, orig_nids, orig_eids):
    """Convert/construct the original node IDs and edge IDs.

    It handles multiple cases:
     * If the graph has been reshuffled and it's a homogeneous graph, we just return
       the original node IDs and edge IDs in the inputs.
     * If the graph has been reshuffled and it's a heterogeneous graph, we need to
       split the original node IDs and edge IDs in the inputs based on the node types
       and edge types.
     * If the graph is not shuffled, the original node IDs and edge IDs don't change.

    Parameters
    ----------
    g : DGLGraph
       The input graph for partitioning.
    sim_g : DGLGraph
        The homogeneous version of the input graph.
    orig_nids : tensor or None
        The original node IDs after the input graph is reshuffled.
    orig_eids : tensor or None
        The original edge IDs after the input graph is reshuffled.

    Returns
    -------
    tensor or dict of tensors, tensor or dict of tensors
    """
    is_hetero = not g.is_homogeneous
    if is_hetero:
        # Get the type IDs
        orig_ntype = F.gather_row(sim_g.ndata[NTYPE], orig_nids)
        orig_etype = F.gather_row(sim_g.edata[ETYPE], orig_eids)
        # Mapping between shuffled global IDs to original per-type IDs
        orig_nids = F.gather_row(sim_g.ndata[NID], orig_nids)
        orig_eids = F.gather_row(sim_g.edata[EID], orig_eids)
        orig_nids = {
            ntype: F.boolean_mask(
                orig_nids, orig_ntype == g.get_ntype_id(ntype)
            )
            for ntype in g.ntypes
        }
        orig_eids = {
            etype: F.boolean_mask(
                orig_eids, orig_etype == g.get_etype_id(etype)
            )
            for etype in g.canonical_etypes
        }
    return orig_nids, orig_eids


def _set_trainer_ids(g, sim_g, node_parts):
    """Set the trainer IDs for each node and edge on the input graph.

    The trainer IDs will be stored as node data and edge data in the input graph.

    Parameters
    ----------
    g : DGLGraph
       The input graph for partitioning.
    sim_g : DGLGraph
        The homogeneous version of the input graph.
    node_parts : tensor
        The node partition ID for each node in `sim_g`.
    """
    if g.is_homogeneous:
        g.ndata["trainer_id"] = node_parts
        # An edge is assigned to a partition based on its destination node.
        g.edata["trainer_id"] = F.gather_row(node_parts, g.edges()[1])
    else:
        for ntype_id, ntype in enumerate(g.ntypes):
            type_idx = sim_g.ndata[NTYPE] == ntype_id
            orig_nid = F.boolean_mask(sim_g.ndata[NID], type_idx)
            trainer_id = F.zeros((len(orig_nid),), F.dtype(node_parts), F.cpu())
            F.scatter_row_inplace(
                trainer_id, orig_nid, F.boolean_mask(node_parts, type_idx)
            )
            g.nodes[ntype].data["trainer_id"] = trainer_id
        for c_etype in g.canonical_etypes:
            # An edge is assigned to a partition based on its destination node.
            _, _, dst_type = c_etype
            trainer_id = F.gather_row(
                g.nodes[dst_type].data["trainer_id"], g.edges(etype=c_etype)[1]
            )
            g.edges[c_etype].data["trainer_id"] = trainer_id


def _partition_to_graphbolt(
    parts,
    part_i,
    part_config,
    part_metadata,
    *,
    store_eids=True,
    store_inner_node=False,
    store_inner_edge=False,
    graph_formats=None,
):
    gpb, _, ntypes, etypes = load_partition_book(
        part_config=part_config, part_id=part_i, part_metadata=part_metadata
    )
    graph = parts[part_i]
    csc_graph = _convert_dgl_partition_to_gb(
        ntypes=ntypes,
        etypes=etypes,
        gpb=gpb,
        part_meta=part_metadata,
        graph=graph,
        store_eids=store_eids,
        store_inner_edge=store_inner_edge,
        store_inner_node=store_inner_node,
        graph_formats=graph_formats,
    )
    rel_path_result = _save_graph_gb(
        part_config=part_config, part_id=part_i, csc_graph=csc_graph
    )
    part_metadata[f"part-{part_i}"]["part_graph_graphbolt"] = rel_path_result


def _update_node_edge_map(node_map_val, edge_map_val, g, num_parts):
    """
    If the original graph contains few nodes or edges for specific node/edge
    types, the partitioned graph may have empty partitions for these types. And
    the node_map_val and edge_map_val will have -1 for the start and end ID of
    these types. This function updates the node_map_val and edge_map_val to be
    contiguous.

    Example case:
    Suppose we have a heterogeneous graph with 3 node/edge types and the number
    of partitions is 3. A possible node_map_val or edge_map_val is as follows:

    | part_id\\Node/Edge Type| Type A |  Type B | Type C |
    |------------------------|--------|---------|--------|
    | 0                      | 0, 1   |  -1, -1 |  2, 3  |
    | 1                      | -1, -1 |  3, 4   |  4, 5  |
    | 2                      | 5, 6   |  7, 8   |  -1, -1|

    As node/edge IDs are contiguous in node/edge type for each partition, we can
    update the node_map_val and edge_map_val via updating the start and end ID
    in row-wise order.

    Updated node_map_val or edge_map_val:

    | part_id\\Node/Edge Type| Type A |  Type B | Type C |
    |------------------------|--------|---------|--------|
    | 0                      |  0, 1  |  1, 1   |  2, 3  |
    | 1                      |  3, 3  |  3, 4   |  4, 5  |
    | 2                      |  5, 6  |  7, 8   |  8, 8  |

    """
    # Update the node_map_val to be contiguous.
    ntype_ids = {ntype: g.get_ntype_id(ntype) for ntype in g.ntypes}
    ntype_ids_reverse = {v: k for k, v in ntype_ids.items()}
    for part_id in range(num_parts):
        for ntype_id in list(ntype_ids.values()):
            ntype = ntype_ids_reverse[ntype_id]
            start_id = node_map_val[ntype][part_id][0]
            end_id = node_map_val[ntype][part_id][1]
            if not (start_id == -1 and end_id == -1):
                continue
            prev_ntype_id = (
                ntype_ids[ntype] - 1
                if ntype_ids[ntype] > 0
                else max(ntype_ids.values())
            )
            prev_ntype = ntype_ids_reverse[prev_ntype_id]
            if ntype_ids[ntype] == 0:
                if part_id == 0:
                    node_map_val[ntype][part_id][0] = 0
                else:
                    node_map_val[ntype][part_id][0] = node_map_val[prev_ntype][
                        part_id - 1
                    ][1]
            else:
                node_map_val[ntype][part_id][0] = node_map_val[prev_ntype][
                    part_id
                ][1]
            node_map_val[ntype][part_id][1] = node_map_val[ntype][part_id][0]
    # Update the edge_map_val to be contiguous.
    etype_ids = {etype: g.get_etype_id(etype) for etype in g.canonical_etypes}
    etype_ids_reverse = {v: k for k, v in etype_ids.items()}
    for part_id in range(num_parts):
        for etype_id in list(etype_ids.values()):
            etype = etype_ids_reverse[etype_id]
            start_id = edge_map_val[etype][part_id][0]
            end_id = edge_map_val[etype][part_id][1]
            if not (start_id == -1 and end_id == -1):
                continue
            prev_etype_id = (
                etype_ids[etype] - 1
                if etype_ids[etype] > 0
                else max(etype_ids.values())
            )
            prev_etype = etype_ids_reverse[prev_etype_id]
            if etype_ids[etype] == 0:
                if part_id == 0:
                    edge_map_val[etype][part_id][0] = 0
                else:
                    edge_map_val[etype][part_id][0] = edge_map_val[prev_etype][
                        part_id - 1
                    ][1]
            else:
                edge_map_val[etype][part_id][0] = edge_map_val[prev_etype][
                    part_id
                ][1]
            edge_map_val[etype][part_id][1] = edge_map_val[etype][part_id][0]


def partition_graph(
    g,
    graph_name,
    num_parts,
    out_path,
    num_hops=1,
    part_method="metis",
    balance_ntypes=None,
    balance_edges=False,
    return_mapping=False,
    num_trainers_per_machine=1,
    objtype="cut",
    graph_formats=None,
    use_graphbolt=False,
    **kwargs,
):
    """Partition a graph for distributed training and store the partitions on files.

    The partitioning occurs in three steps: 1) run a partition algorithm (e.g., Metis) to
    assign nodes to partitions; 2) construct partition graph structure based on
    the node assignment; 3) split the node features and edge features based on
    the partition result.

    When a graph is partitioned, each partition can contain *HALO* nodes, which are assigned
    to other partitions but are included in this partition for efficiency purpose.
    In this document, *local nodes/edges* refers to the nodes and edges that truly belong to
    a partition. The rest are "HALO nodes/edges".

    The partitioned data is stored into multiple files organized as follows:

    .. code-block:: none

        data_root_dir/
          |-- graph_name.json     # partition configuration file in JSON
          |-- node_map.npy        # partition id of each node stored in a numpy array (optional)
          |-- edge_map.npy        # partition id of each edge stored in a numpy array (optional)
          |-- part0/              # data for partition 0
              |-- node_feats.dgl  # node features stored in binary format
              |-- edge_feats.dgl  # edge features stored in binary format
              |-- graph.dgl       # graph structure of this partition stored in binary format
          |-- part1/              # data for partition 1
              |-- node_feats.dgl
              |-- edge_feats.dgl
              |-- graph.dgl

    First, the metadata of the original graph and the partitioning is stored in a JSON file
    named after ``graph_name``. This JSON file contains the information of the original graph
    as well as the path of the files that store each partition. Below show an example.

    .. code-block:: none

        {
           "graph_name" : "test",
           "part_method" : "metis",
           "num_parts" : 2,
           "halo_hops" : 1,
           "node_map": {
               "_N": [ [ 0, 1261310 ],
                       [ 1261310, 2449029 ] ]
           },
           "edge_map": {
               "_N:_E:_N": [ [ 0, 62539528 ],
                             [ 62539528, 123718280 ] ]
           },
           "etypes": { "_N:_E:_N": 0 },
           "ntypes": { "_N": 0 },
           "num_nodes" : 1000000,
           "num_edges" : 52000000,
           "part-0" : {
             "node_feats" : "data_root_dir/part0/node_feats.dgl",
             "edge_feats" : "data_root_dir/part0/edge_feats.dgl",
             "part_graph" : "data_root_dir/part0/graph.dgl",
           },
           "part-1" : {
             "node_feats" : "data_root_dir/part1/node_feats.dgl",
             "edge_feats" : "data_root_dir/part1/edge_feats.dgl",
             "part_graph" : "data_root_dir/part1/graph.dgl",
           },
        }

    Here are the definition of the fields in the partition configuration file:

    * ``graph_name`` is the name of the graph given by a user.
    * ``part_method`` is the method used to assign nodes to partitions.
      Currently, it supports "random" and "metis".
    * ``num_parts`` is the number of partitions.
    * ``halo_hops`` is the number of hops of nodes we include in a partition as HALO nodes.
    * ``node_map`` is the node assignment map, which tells the partition ID a node is assigned to.
      The format of ``node_map`` is described below.
    * ``edge_map`` is the edge assignment map, which tells the partition ID an edge is assigned to.
    * ``num_nodes`` is the number of nodes in the global graph.
    * ``num_edges`` is the number of edges in the global graph.
    * `part-*` stores the data of a partition.

    As node/edge IDs are reshuffled, ``node_map`` and ``edge_map`` contains the information
    for mapping between global node/edge IDs to partition-local node/edge IDs.
    For heterogeneous graphs, the information in ``node_map`` and ``edge_map`` can also be used
    to compute node types and edge types. The format of the data in ``node_map`` and ``edge_map``
    is as follows:

    .. code-block:: none

        {
            "node_type": [ [ part1_start, part1_end ],
                           [ part2_start, part2_end ],
                           ... ],
            ...
        },

    Essentially, ``node_map`` and ``edge_map`` are dictionaries. The keys are
    node etypes and canonical edge types respectively. The values are lists of pairs
    containing the start and end of the ID range for the corresponding types in a partition.
    The length of the list is the number of
    partitions; each element in the list is a tuple that stores the start and the end of
    an ID range for a particular node/edge type in the partition.

    The graph structure of a partition is stored in a file with the DGLGraph format.
    Nodes in each partition is *relabeled* to always start with zero. We call the node
    ID in the original graph, *global ID*, while the relabeled ID in each partition,
    *local ID*. Each partition graph has an integer node data tensor stored under name
    `dgl.NID` and each value is the node's global ID. Similarly, edges are relabeled too
    and the mapping from local ID to global ID is stored as an integer edge data tensor
    under name `dgl.EID`. For a heterogeneous graph, the DGLGraph also contains a node
    data `dgl.NTYPE` for node type and an edge data `dgl.ETYPE` for the edge type.

    The partition graph contains additional node data ("inner_node") and
    edge data ("inner_edge"):

    * "inner_node" indicates whether a node belongs to a partition.
    * "inner_edge" indicates whether an edge belongs to a partition.

    Node and edge features are splitted and stored together with each graph partition.
    All node/edge features in a partition are stored in a file with DGL format. The node/edge
    features are stored in dictionaries, in which the key is the node/edge data name and
    the value is a tensor. We do not store features of HALO nodes and edges.

    When performing Metis partitioning, we can put some constraint on the partitioning.
    Current, it supports two constrants to balance the partitioning. By default, Metis
    always tries to balance the number of nodes in each partition.

    * ``balance_ntypes`` balances the number of nodes of different types in each partition.
    * ``balance_edges`` balances the number of edges in each partition.

    To balance the node types, a user needs to pass a vector of N elements to indicate
    the type of each node. N is the number of nodes in the input graph.

    Parameters
    ----------
    g : DGLGraph
        The input graph to partition
    graph_name : str
        The name of the graph. The name will be used to construct
        :py:meth:`~dgl.distributed.DistGraph`.
    num_parts : int
        The number of partitions
    out_path : str
        The path to store the files for all partitioned data.
    num_hops : int, optional
        The number of hops of HALO nodes we construct on a partition graph structure.
        The default value is 1.
    part_method : str, optional
        The partition method. It supports "random" and "metis". The default value is "metis".
    balance_ntypes : tensor, optional
        Node type of each node. This is a 1D-array of integers. Its values indicates the node
        type of each node. This argument is used by Metis partition. When the argument is
        specified, the Metis algorithm will try to partition the input graph into partitions where
        each partition has roughly the same number of nodes for each node type. The default value
        is None, which means Metis partitions the graph to only balance the number of nodes.
    balance_edges : bool
        Indicate whether to balance the edges in each partition. This argument is used by
        the Metis algorithm.
    return_mapping : bool
        Indicate whether to return the mapping between shuffled node/edge IDs and the original
        node/edge IDs.
    num_trainers_per_machine : int, optional
        The number of trainers per machine. If is not 1, the whole graph will be first partitioned
        to each trainer, that is num_parts*num_trainers_per_machine parts. And the trainer ids of
        each node will be stored in the node feature 'trainer_id'. Then the partitions of trainers
        on the same machine will be coalesced into one larger partition. The final number of
        partitions is `num_part`.
    objtype : str, "cut" or "vol"
        Set the objective as edge-cut minimization or communication volume minimization. This
        argument is used by the Metis algorithm.
    graph_formats : str or list[str]
        Save partitions in specified formats. It could be any combination of ``coo``,
        ``csc`` and ``csr``. If not specified, save one format only according to what
        format is available. If multiple formats are available, selection priority
        from high to low is ``coo``, ``csc``, ``csr``.
    use_graphbolt : bool, optional
        Whether to save partitions in GraphBolt format. Default: False.
    kwargs : dict
        Other keyword arguments for converting DGL partitions to GraphBolt.

    Returns
    -------
    Tensor or dict of tensors, optional
        If `return_mapping=True`, return a 1D tensor that indicates the mapping between shuffled
        node IDs and the original node IDs for a homogeneous graph; return a dict of 1D tensors
        whose key is the node type and value is a 1D tensor mapping between shuffled node IDs and
        the original node IDs for each node type for a heterogeneous graph.
    Tensor or dict of tensors, optional
        If `return_mapping=True`, return a 1D tensor that indicates the mapping between shuffled
        edge IDs and the original edge IDs for a homogeneous graph; return a dict of 1D tensors
        whose key is the edge type and value is a 1D tensor mapping between shuffled edge IDs and
        the original edge IDs for each edge type for a heterogeneous graph.

    Examples
    --------
    >>> dgl.distributed.partition_graph(g, 'test', 4, num_hops=1, part_method='metis',
    ...                                 out_path='output/',
    ...                                 balance_ntypes=g.ndata['train_mask'],
    ...                                 balance_edges=True)
    >>> (
    ...     g, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,
    ... ) = dgl.distributed.load_partition('output/test.json', 0)
    """
    # 'coo' is required for partition
    assert "coo" in np.concatenate(
        list(g.formats().values())
    ), "'coo' format should be allowed for partitioning graph."

    def get_homogeneous(g, balance_ntypes):
        if g.is_homogeneous:
            sim_g = to_homogeneous(g)
            if isinstance(balance_ntypes, dict):
                assert len(balance_ntypes) == 1
                bal_ntypes = list(balance_ntypes.values())[0]
            else:
                bal_ntypes = balance_ntypes
        elif isinstance(balance_ntypes, dict):
            # Here we assign node types for load balancing.
            # The new node types includes the ones provided by users.
            num_ntypes = 0
            for key in g.ntypes:
                if key in balance_ntypes:
                    g.nodes[key].data["bal_ntype"] = (
                        F.astype(balance_ntypes[key], F.int32) + num_ntypes
                    )
                    uniq_ntypes = F.unique(balance_ntypes[key])
                    assert np.all(
                        F.asnumpy(uniq_ntypes) == np.arange(len(uniq_ntypes))
                    )
                    num_ntypes += len(uniq_ntypes)
                else:
                    g.nodes[key].data["bal_ntype"] = (
                        F.ones((g.num_nodes(key),), F.int32, F.cpu())
                        * num_ntypes
                    )
                    num_ntypes += 1
            sim_g = to_homogeneous(g, ndata=["bal_ntype"])
            bal_ntypes = sim_g.ndata["bal_ntype"]
            print(
                "The graph has {} node types and balance among {} types".format(
                    len(g.ntypes), len(F.unique(bal_ntypes))
                )
            )
            # We now no longer need them.
            for key in g.ntypes:
                del g.nodes[key].data["bal_ntype"]
            del sim_g.ndata["bal_ntype"]
        else:
            sim_g = to_homogeneous(g)
            bal_ntypes = sim_g.ndata[NTYPE]
        return sim_g, bal_ntypes

    if objtype not in ["cut", "vol"]:
        raise ValueError

    if num_parts == 1:
        start = time.time()
        sim_g, balance_ntypes = get_homogeneous(g, balance_ntypes)
        print(
            "Converting to homogeneous graph takes {:.3f}s, peak mem: {:.3f} GB".format(
                time.time() - start, get_peak_mem()
            )
        )
        assert num_trainers_per_machine >= 1
        if num_trainers_per_machine > 1:
            # First partition the whole graph to each trainer and save the trainer ids in
            # the node feature "trainer_id".
            start = time.time()
            node_parts = metis_partition_assignment(
                sim_g,
                num_parts * num_trainers_per_machine,
                balance_ntypes=balance_ntypes,
                balance_edges=balance_edges,
                mode="k-way",
            )
            _set_trainer_ids(g, sim_g, node_parts)
            print(
                "Assigning nodes to METIS partitions takes {:.3f}s, peak mem: {:.3f} GB".format(
                    time.time() - start, get_peak_mem()
                )
            )

        node_parts = F.zeros((sim_g.num_nodes(),), F.int64, F.cpu())
        parts = {0: sim_g.clone()}
        orig_nids = parts[0].ndata[NID] = F.arange(0, sim_g.num_nodes())
        orig_eids = parts[0].edata[EID] = F.arange(0, sim_g.num_edges())
        # For one partition, we don't really shuffle nodes and edges. We just need to simulate
        # it and set node data and edge data of orig_id.
        parts[0].ndata["orig_id"] = orig_nids
        parts[0].edata["orig_id"] = orig_eids
        if return_mapping:
            if g.is_homogeneous:
                orig_nids = F.arange(0, sim_g.num_nodes())
                orig_eids = F.arange(0, sim_g.num_edges())
            else:
                orig_nids = {
                    ntype: F.arange(0, g.num_nodes(ntype)) for ntype in g.ntypes
                }
                orig_eids = {
                    etype: F.arange(0, g.num_edges(etype))
                    for etype in g.canonical_etypes
                }
        parts[0].ndata["inner_node"] = F.ones(
            (sim_g.num_nodes(),),
            RESERVED_FIELD_DTYPE["inner_node"],
            F.cpu(),
        )
        parts[0].edata["inner_edge"] = F.ones(
            (sim_g.num_edges(),),
            RESERVED_FIELD_DTYPE["inner_edge"],
            F.cpu(),
        )
    elif part_method in ("metis", "random"):
        start = time.time()
        sim_g, balance_ntypes = get_homogeneous(g, balance_ntypes)
        print(
            "Converting to homogeneous graph takes {:.3f}s, peak mem: {:.3f} GB".format(
                time.time() - start, get_peak_mem()
            )
        )
        if part_method == "metis":
            assert num_trainers_per_machine >= 1
            start = time.time()
            if num_trainers_per_machine > 1:
                # First partition the whole graph to each trainer and save the trainer ids in
                # the node feature "trainer_id".
                node_parts = metis_partition_assignment(
                    sim_g,
                    num_parts * num_trainers_per_machine,
                    balance_ntypes=balance_ntypes,
                    balance_edges=balance_edges,
                    mode="k-way",
                    objtype=objtype,
                )
                _set_trainer_ids(g, sim_g, node_parts)

                # And then coalesce the partitions of trainers on the same machine into one
                # larger partition.
                node_parts = F.floor_div(node_parts, num_trainers_per_machine)
            else:
                node_parts = metis_partition_assignment(
                    sim_g,
                    num_parts,
                    balance_ntypes=balance_ntypes,
                    balance_edges=balance_edges,
                    objtype=objtype,
                )
            print(
                "Assigning nodes to METIS partitions takes {:.3f}s, peak mem: {:.3f} GB".format(
                    time.time() - start, get_peak_mem()
                )
            )
        else:
            node_parts = random_choice(num_parts, sim_g.num_nodes())
        start = time.time()
        parts, orig_nids, orig_eids = partition_graph_with_halo(
            sim_g, node_parts, num_hops, reshuffle=True
        )
        print(
            "Splitting the graph into partitions takes {:.3f}s, peak mem: {:.3f} GB".format(
                time.time() - start, get_peak_mem()
            )
        )
        if return_mapping:
            orig_nids, orig_eids = _get_orig_ids(g, sim_g, orig_nids, orig_eids)
    else:
        raise Exception("Unknown partitioning method: " + part_method)

    # If the input is a heterogeneous graph, get the original node types and original node IDs.
    # `part' has three types of node data at this point.
    # NTYPE: the node type.
    # orig_id: the global node IDs in the homogeneous version of input graph.
    # NID: the global node IDs in the reshuffled homogeneous version of the input graph.
    if not g.is_homogeneous:
        for name in parts:
            orig_ids = parts[name].ndata["orig_id"]
            ntype = F.gather_row(sim_g.ndata[NTYPE], orig_ids)
            parts[name].ndata[NTYPE] = F.astype(
                ntype, RESERVED_FIELD_DTYPE[NTYPE]
            )
            assert np.all(
                F.asnumpy(ntype) == F.asnumpy(parts[name].ndata[NTYPE])
            )
            # Get the original edge types and original edge IDs.
            orig_ids = parts[name].edata["orig_id"]
            etype = F.gather_row(sim_g.edata[ETYPE], orig_ids)
            parts[name].edata[ETYPE] = F.astype(
                etype, RESERVED_FIELD_DTYPE[ETYPE]
            )
            assert np.all(
                F.asnumpy(etype) == F.asnumpy(parts[name].edata[ETYPE])
            )

            # Calculate the global node IDs to per-node IDs mapping.
            inner_ntype = F.boolean_mask(
                parts[name].ndata[NTYPE], parts[name].ndata["inner_node"] == 1
            )
            inner_nids = F.boolean_mask(
                parts[name].ndata[NID], parts[name].ndata["inner_node"] == 1
            )
            for ntype in g.ntypes:
                inner_ntype_mask = inner_ntype == g.get_ntype_id(ntype)
                if F.sum(F.astype(inner_ntype_mask, F.int64), 0) == 0:
                    # Skip if there is no node of this type in this partition.
                    continue
                typed_nids = F.boolean_mask(inner_nids, inner_ntype_mask)
                # inner node IDs are in a contiguous ID range.
                expected_range = np.arange(
                    int(F.as_scalar(typed_nids[0])),
                    int(F.as_scalar(typed_nids[-1])) + 1,
                )
                assert np.all(F.asnumpy(typed_nids) == expected_range)
            # Calculate the global edge IDs to per-edge IDs mapping.
            inner_etype = F.boolean_mask(
                parts[name].edata[ETYPE], parts[name].edata["inner_edge"] == 1
            )
            inner_eids = F.boolean_mask(
                parts[name].edata[EID], parts[name].edata["inner_edge"] == 1
            )
            for etype in g.canonical_etypes:
                inner_etype_mask = inner_etype == g.get_etype_id(etype)
                if F.sum(F.astype(inner_etype_mask, F.int64), 0) == 0:
                    # Skip if there is no edge of this type in this partition.
                    continue
                typed_eids = np.sort(
                    F.asnumpy(F.boolean_mask(inner_eids, inner_etype_mask))
                )
                assert np.all(
                    typed_eids
                    == np.arange(int(typed_eids[0]), int(typed_eids[-1]) + 1)
                )

    os.makedirs(out_path, mode=0o775, exist_ok=True)
    tot_num_inner_edges = 0
    out_path = os.path.abspath(out_path)

    # With reshuffling, we can ensure that all nodes and edges are reshuffled
    # and are in contiguous ID space.
    if num_parts > 1:
        node_map_val = {}
        edge_map_val = {}
        for ntype in g.ntypes:
            ntype_id = g.get_ntype_id(ntype)
            val = []
            node_map_val[ntype] = []
            for i in parts:
                inner_node_mask = _get_inner_node_mask(parts[i], ntype_id)
                val.append(
                    F.as_scalar(F.sum(F.astype(inner_node_mask, F.int64), 0))
                )
                if F.sum(F.astype(inner_node_mask, F.int64), 0) == 0:
                    node_map_val[ntype].append([-1, -1])
                    continue
                inner_nids = F.boolean_mask(
                    parts[i].ndata[NID], inner_node_mask
                )
                node_map_val[ntype].append(
                    [
                        int(F.as_scalar(inner_nids[0])),
                        int(F.as_scalar(inner_nids[-1])) + 1,
                    ]
                )
            val = np.cumsum(val).tolist()
            assert val[-1] == g.num_nodes(ntype)
        for etype in g.canonical_etypes:
            etype_id = g.get_etype_id(etype)
            val = []
            edge_map_val[etype] = []
            for i in parts:
                inner_edge_mask = _get_inner_edge_mask(parts[i], etype_id)
                val.append(
                    F.as_scalar(F.sum(F.astype(inner_edge_mask, F.int64), 0))
                )
                if F.sum(F.astype(inner_edge_mask, F.int64), 0) == 0:
                    edge_map_val[etype].append([-1, -1])
                    continue
                inner_eids = np.sort(
                    F.asnumpy(
                        F.boolean_mask(parts[i].edata[EID], inner_edge_mask)
                    )
                )
                edge_map_val[etype].append(
                    [int(inner_eids[0]), int(inner_eids[-1]) + 1]
                )
            val = np.cumsum(val).tolist()
            assert val[-1] == g.num_edges(etype)
        # Update the node_map_val and edge_map_val to be contiguous.
        _update_node_edge_map(node_map_val, edge_map_val, g, num_parts)
    else:
        node_map_val = {}
        edge_map_val = {}
        for ntype in g.ntypes:
            ntype_id = g.get_ntype_id(ntype)
            inner_node_mask = _get_inner_node_mask(parts[0], ntype_id)
            inner_nids = F.boolean_mask(parts[0].ndata[NID], inner_node_mask)
            node_map_val[ntype] = [
                [
                    int(F.as_scalar(inner_nids[0])),
                    int(F.as_scalar(inner_nids[-1])) + 1,
                ]
            ]
        for etype in g.canonical_etypes:
            etype_id = g.get_etype_id(etype)
            inner_edge_mask = _get_inner_edge_mask(parts[0], etype_id)
            inner_eids = F.boolean_mask(parts[0].edata[EID], inner_edge_mask)
            edge_map_val[etype] = [
                [
                    int(F.as_scalar(inner_eids[0])),
                    int(F.as_scalar(inner_eids[-1])) + 1,
                ]
            ]

        # Double check that the node IDs in the global ID space are sorted.
        for ntype in node_map_val:
            val = np.concatenate([np.array(l) for l in node_map_val[ntype]])
            assert np.all(val[:-1] <= val[1:])
        for etype in edge_map_val:
            val = np.concatenate([np.array(l) for l in edge_map_val[etype]])
            assert np.all(val[:-1] <= val[1:])

    start = time.time()
    ntypes = {ntype: g.get_ntype_id(ntype) for ntype in g.ntypes}
    etypes = {etype: g.get_etype_id(etype) for etype in g.canonical_etypes}
    part_metadata = {
        "graph_name": graph_name,
        "num_nodes": g.num_nodes(),
        "num_edges": g.num_edges(),
        "part_method": part_method,
        "num_parts": num_parts,
        "halo_hops": num_hops,
        "node_map": node_map_val,
        "edge_map": edge_map_val,
        "ntypes": ntypes,
        "etypes": etypes,
    }
    part_config = os.path.join(out_path, graph_name + ".json")
    for part_id in range(num_parts):
        part = parts[part_id]

        # Get the node/edge features of each partition.
        node_feats = {}
        edge_feats = {}
        if num_parts > 1:
            for ntype in g.ntypes:
                ntype_id = g.get_ntype_id(ntype)
                # To get the edges in the input graph, we should use original node IDs.
                # Both orig_id and NID stores the per-node-type IDs.
                ndata_name = "orig_id"
                inner_node_mask = _get_inner_node_mask(part, ntype_id)
                # This is global node IDs.
                local_nodes = F.boolean_mask(
                    part.ndata[ndata_name], inner_node_mask
                )
                if len(g.ntypes) > 1:
                    # If the input is a heterogeneous graph.
                    local_nodes = F.gather_row(sim_g.ndata[NID], local_nodes)
                    print(
                        "part {} has {} nodes of type {} and {} are inside the partition".format(
                            part_id,
                            F.as_scalar(
                                F.sum(part.ndata[NTYPE] == ntype_id, 0)
                            ),
                            ntype,
                            len(local_nodes),
                        )
                    )
                else:
                    print(
                        "part {} has {} nodes and {} are inside the partition".format(
                            part_id, part.num_nodes(), len(local_nodes)
                        )
                    )

                for name in g.nodes[ntype].data:
                    if name in [NID, "inner_node"]:
                        continue
                    node_feats[ntype + "/" + name] = F.gather_row(
                        g.nodes[ntype].data[name], local_nodes
                    )

            for etype in g.canonical_etypes:
                etype_id = g.get_etype_id(etype)
                edata_name = "orig_id"
                inner_edge_mask = _get_inner_edge_mask(part, etype_id)
                # This is global edge IDs.
                local_edges = F.boolean_mask(
                    part.edata[edata_name], inner_edge_mask
                )
                if not g.is_homogeneous:
                    local_edges = F.gather_row(sim_g.edata[EID], local_edges)
                    print(
                        "part {} has {} edges of type {} and {} are inside the partition".format(
                            part_id,
                            F.as_scalar(
                                F.sum(part.edata[ETYPE] == etype_id, 0)
                            ),
                            etype,
                            len(local_edges),
                        )
                    )
                else:
                    print(
                        "part {} has {} edges and {} are inside the partition".format(
                            part_id, part.num_edges(), len(local_edges)
                        )
                    )
                tot_num_inner_edges += len(local_edges)

                for name in g.edges[etype].data:
                    if name in [EID, "inner_edge"]:
                        continue
                    edge_feats[
                        _etype_tuple_to_str(etype) + "/" + name
                    ] = F.gather_row(g.edges[etype].data[name], local_edges)
        else:
            for ntype in g.ntypes:
                if len(g.ntypes) > 1:
                    ndata_name = "orig_id"
                    ntype_id = g.get_ntype_id(ntype)
                    inner_node_mask = _get_inner_node_mask(part, ntype_id)
                    # This is global node IDs.
                    local_nodes = F.boolean_mask(
                        part.ndata[ndata_name], inner_node_mask
                    )
                    local_nodes = F.gather_row(sim_g.ndata[NID], local_nodes)
                else:
                    local_nodes = sim_g.ndata[NID]
                for name in g.nodes[ntype].data:
                    if name in [NID, "inner_node"]:
                        continue
                    node_feats[ntype + "/" + name] = F.gather_row(
                        g.nodes[ntype].data[name], local_nodes
                    )
            for etype in g.canonical_etypes:
                if not g.is_homogeneous:
                    edata_name = "orig_id"
                    etype_id = g.get_etype_id(etype)
                    inner_edge_mask = _get_inner_edge_mask(part, etype_id)
                    # This is global edge IDs.
                    local_edges = F.boolean_mask(
                        part.edata[edata_name], inner_edge_mask
                    )
                    local_edges = F.gather_row(sim_g.edata[EID], local_edges)
                else:
                    local_edges = sim_g.edata[EID]
                for name in g.edges[etype].data:
                    if name in [EID, "inner_edge"]:
                        continue
                    edge_feats[
                        _etype_tuple_to_str(etype) + "/" + name
                    ] = F.gather_row(g.edges[etype].data[name], local_edges)
        # delete `orig_id` from ndata/edata
        del part.ndata["orig_id"]
        del part.edata["orig_id"]

        part_dir = os.path.join(out_path, "part" + str(part_id))
        node_feat_file = os.path.join(part_dir, "node_feat.dgl")
        edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")

        os.makedirs(part_dir, mode=0o775, exist_ok=True)
        save_tensors(node_feat_file, node_feats)
        save_tensors(edge_feat_file, edge_feats)

        part_metadata["part-{}".format(part_id)] = {
            "node_feats": os.path.relpath(node_feat_file, out_path),
            "edge_feats": os.path.relpath(edge_feat_file, out_path),
        }
        sort_etypes = len(g.etypes) > 1
        part = process_partitions(part, graph_formats, sort_etypes)

    # transmit to graphbolt and save graph
    if use_graphbolt:
        # save FusedCSCSamplingGraph
        kwargs["graph_formats"] = graph_formats
        n_jobs = kwargs.pop("n_jobs", 1)
        mp_ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(  # pylint: disable=unexpected-keyword-arg
            max_workers=min(num_parts, n_jobs),
            mp_context=mp_ctx,
        ) as executor:
            for part_id in range(num_parts):
                executor.submit(
                    _partition_to_graphbolt(
                        part_i=part_id,
                        part_config=part_config,
                        part_metadata=part_metadata,
                        parts=parts,
                        **kwargs,
                    )
                )
        part_metadata["node_map_dtype"] = "int64"
        part_metadata["edge_map_dtype"] = "int64"
    else:
        for part_id, part in parts.items():
            part_dir = os.path.join(out_path, "part" + str(part_id))
            part_graph_file = os.path.join(part_dir, "graph.dgl")
            part_metadata["part-{}".format(part_id)][
                "part_graph"
            ] = os.path.relpath(part_graph_file, out_path)
            # save DGLGraph
            _save_dgl_graphs(
                part_graph_file,
                [part],
                formats=graph_formats,
            )

    _dump_part_config(part_config, part_metadata)

    num_cuts = sim_g.num_edges() - tot_num_inner_edges
    if num_parts == 1:
        num_cuts = 0
    print(
        "There are {} edges in the graph and {} edge cuts for {} partitions.".format(
            g.num_edges(), num_cuts, num_parts
        )
    )

    print(
        "Save partitions: {:.3f} seconds, peak memory: {:.3f} GB".format(
            time.time() - start, get_peak_mem()
        )
    )

    if return_mapping:
        return orig_nids, orig_eids


# [TODO][Rui] Due to int64_t is expected in RPC, we have to limit the data type
# of node/edge IDs to int64_t. See more details in #7175.
DTYPES_TO_CHECK = {
    "default": [torch.int32, torch.int64],
    NID: [torch.int64],
    EID: [torch.int64],
    NTYPE: [torch.int8, torch.int16, torch.int32, torch.int64],
    ETYPE: [torch.int8, torch.int16, torch.int32, torch.int64],
    "inner_node": [torch.uint8],
    "inner_edge": [torch.uint8],
    "part_id": [torch.int8, torch.int16, torch.int32, torch.int64],
}


def _cast_to_minimum_dtype(predicate, data, field=None):
    if data is None:
        return data
    dtypes_to_check = DTYPES_TO_CHECK.get(field, DTYPES_TO_CHECK["default"])
    if data.dtype not in dtypes_to_check:
        dgl_warning(
            f"Skipping as the data type of field {field} is {data.dtype}, "
            f"while supported data types are {dtypes_to_check}."
        )
        return data
    for dtype in dtypes_to_check:
        if predicate < torch.iinfo(dtype).max:
            return data.to(dtype)
    return data


# Utility functions.
def is_homogeneous(ntypes, etypes):
    """Checks if the provided ntypes and etypes form a homogeneous graph."""
    return len(ntypes) == 1 and len(etypes) == 1


def init_type_per_edge(graph, gpb):
    """Initialize edge ids for every edge type."""
    etype_ids = gpb.map_to_per_etype(graph.edata[EID])[0]
    return etype_ids


def _load_part(part_config, part_id, parts=None):
    """load parts from variable or dist."""
    if parts is None:
        graph, _, _, _, _, _, _ = load_partition(
            part_config, part_id, load_feats=False
        )
    else:
        graph = parts[part_id]
    return graph


def _save_graph_gb(part_config, part_id, csc_graph):
    csc_graph_save_dir = os.path.join(
        os.path.dirname(part_config),
        f"part{part_id}",
    )
    csc_graph_path = os.path.join(
        csc_graph_save_dir, "fused_csc_sampling_graph.pt"
    )
    torch.save(csc_graph, csc_graph_path)

    return os.path.relpath(csc_graph_path, os.path.dirname(part_config))


def cast_various_to_minimum_dtype_gb(
    num_parts,
    indptr,
    indices,
    type_per_edge,
    etypes,
    ntypes,
    node_attributes,
    edge_attributes,
    part_meta=None,
    graph=None,
    edge_count=None,
    node_count=None,
    tot_edge_count=None,
    tot_node_count=None,
):
    """Cast various data to minimum dtype."""
    if graph is not None:
        assert part_meta is not None
        tot_edge_count = graph.num_edges()
        tot_node_count = graph.num_nodes()
        node_count = part_meta["num_nodes"]
        edge_count = part_meta["num_edges"]
    else:
        assert tot_edge_count is not None
        assert tot_node_count is not None
        assert edge_count is not None
        assert node_count is not None

    # Cast 1: indptr.
    indptr = _cast_to_minimum_dtype(tot_edge_count, indptr)
    # Cast 2: indices.
    indices = _cast_to_minimum_dtype(tot_node_count, indices)
    # Cast 3: type_per_edge.
    type_per_edge = _cast_to_minimum_dtype(
        len(etypes), type_per_edge, field=ETYPE
    )
    # Cast 4: node/edge_attributes.
    predicates = {
        NID: node_count,
        "part_id": num_parts,
        NTYPE: len(ntypes),
        EID: edge_count,
        ETYPE: len(etypes),
        DGL2GB_EID: edge_count,
        GB_DST_ID: node_count,
    }
    for attributes in [node_attributes, edge_attributes]:
        for key in attributes:
            if key not in predicates:
                continue
            attributes[key] = _cast_to_minimum_dtype(
                predicates[key], attributes[key], field=key
            )
    return indptr, indices, type_per_edge


def _create_attributes_gb(
    graph,
    gpb,
    edge_ids,
    is_homo,
    store_inner_node,
    store_inner_edge,
    store_eids,
    debug_mode,
):
    # Save node attributes. Detailed attributes are shown below.
    #  DGL_GB\Attributes  dgl.NID("_ID")  dgl.NTYPE("_TYPE")  "inner_node"  "part_id"
    #  DGL_Homograph           ✅                🚫                  ✅          ✅
    #  GB_Homograph            ✅                🚫               optional       🚫
    #  DGL_Heterograph         ✅                ✅                  ✅          ✅
    #  GB_Heterograph          ✅                🚫               optional       🚫
    required_node_attrs = [NID]
    if store_inner_node:
        required_node_attrs.append("inner_node")
    if debug_mode:
        required_node_attrs = list(graph.ndata.keys())
    node_attributes = {attr: graph.ndata[attr] for attr in required_node_attrs}

    # Save edge attributes. Detailed attributes are shown below.
    #  DGL_GB\Attributes  dgl.EID("_ID")  dgl.ETYPE("_TYPE")  "inner_edge"
    #  DGL_Homograph           ✅               🚫                  ✅
    #  GB_Homograph         optional            🚫               optional
    #  DGL_Heterograph         ✅               ✅                  ✅
    #  GB_Heterograph       optional            ✅               optional
    type_per_edge = None
    if not is_homo:
        type_per_edge = init_type_per_edge(graph, gpb)[edge_ids]
        type_per_edge = type_per_edge.to(RESERVED_FIELD_DTYPE[ETYPE])
    required_edge_attrs = []
    if store_eids:
        required_edge_attrs.append(EID)
    if store_inner_edge:
        required_edge_attrs.append("inner_edge")
    if debug_mode:
        required_edge_attrs = list(graph.edata.keys())
    edge_attributes = {
        attr: graph.edata[attr][edge_ids] for attr in required_edge_attrs
    }
    return node_attributes, edge_attributes, type_per_edge


def _convert_dgl_partition_to_gb(
    ntypes,
    etypes,
    gpb,
    part_meta,
    graph,
    graph_formats=None,
    store_eids=False,
    store_inner_node=False,
    store_inner_edge=False,
):
    """Converts a single DGL partition to GraphBolt.

    Parameters
    ----------
    node types : dict
        The node types
    edge types : dict
        The edge types
    gpb : GraphPartitionBook
        The global partition information.
    part_meta : dict
        Contain the meta data of the partition.
    graph : DGLGraph
        The graph to be converted to graphbolt graph.
    graph_formats : str or list[str], optional
        Save partitions in specified formats. It could be any combination of
        `coo`, `csc`. As `csc` format is mandatory for `FusedCSCSamplingGraph`,
        it is not necessary to specify this argument. It's mainly for
        specifying `coo` format to save edge ID mapping and destination node
        IDs. If not specified, whether to save `coo` format is determined by
        the availability of the format in DGL partitions. Default: None.
    store_eids : bool, optional
        Whether to store edge IDs in the new graph. Default: True.
    store_inner_node : bool, optional
        Whether to store inner node mask in the new graph. Default: False.
    store_inner_edge : bool, optional
        Whether to store inner edge mask in the new graph. Default: False.
    """
    debug_mode = "DGL_DIST_DEBUG" in os.environ
    if debug_mode:
        dgl_warning(
            "Running in debug mode which means all attributes of DGL partitions"
            " will be saved to the new format."
        )
    num_parts = part_meta["num_parts"]

    is_homo = is_homogeneous(ntypes, etypes)
    node_type_to_id = (
        None if is_homo else {ntype: ntid for ntid, ntype in enumerate(ntypes)}
    )
    edge_type_to_id = (
        None
        if is_homo
        else {
            gb.etype_tuple_to_str(etype): etid for etype, etid in etypes.items()
        }
    )
    # Obtain CSC indtpr and indices.
    indptr, indices, edge_ids = graph.adj_tensors("csc")

    node_attributes, edge_attributes, type_per_edge = _create_attributes_gb(
        graph,
        gpb,
        edge_ids,
        is_homo,
        store_inner_node,
        store_inner_edge,
        store_eids,
        debug_mode,
    )
    # When converting DGLGraph to FusedCSCSamplingGraph, edge IDs are
    # re-ordered(actually FusedCSCSamplingGraph does not have edge IDs
    # in nature). So we need to save such re-order info for any
    # operations that uses original local edge IDs. For now, this is
    # required by `DistGraph.find_edges()` for link prediction tasks.
    #
    # What's more, in order to find the dst nodes efficiently, we save
    # dst nodes directly in the edge attributes.
    #
    # So we require additional `(2 * E) * dtype` space in total.
    if graph_formats is not None and isinstance(graph_formats, str):
        graph_formats = [graph_formats]
    save_coo = (
        graph_formats is None and "coo" in graph.formats()["created"]
    ) or (graph_formats is not None and "coo" in graph_formats)
    if save_coo:
        edge_attributes[DGL2GB_EID] = torch.argsort(edge_ids)
        edge_attributes[GB_DST_ID] = gb.expand_indptr(
            indptr, dtype=indices.dtype
        )

    indptr, indices, type_per_edge = cast_various_to_minimum_dtype_gb(
        graph=graph,
        part_meta=part_meta,
        num_parts=num_parts,
        indptr=indptr,
        indices=indices,
        type_per_edge=type_per_edge,
        etypes=etypes,
        ntypes=ntypes,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    )

    csc_graph = gb.fused_csc_sampling_graph(
        indptr,
        indices,
        node_type_offset=None,
        type_per_edge=type_per_edge,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
        node_type_to_id=node_type_to_id,
        edge_type_to_id=edge_type_to_id,
    )
    return csc_graph


def gb_convert_single_dgl_partition(
    part_id,
    graph_formats,
    part_config,
    store_eids=True,
    store_inner_node=True,
    store_inner_edge=True,
):
    """
    The pipeline converting signle partition to graphbolt.

    Parameters
    ----------
    part_id : int
        The partition ID.
    graph_formats : str or list[str]
        Save partitions in specified formats. It could be any combination of
        `coo`, `csc`. As `csc` format is mandatory for `FusedCSCSamplingGraph`,
        it is not necessary to specify this argument. It's mainly for
        specifying `coo` format to save edge ID mapping and destination node
        IDs. If not specified, whether to save `coo` format is determined by
        the availability of the format in DGL partitions. Default: None.
    part_config : str
        The path of the partition config file.
    store_eids : bool, optional
        Whether to store edge IDs in the new graph. Default: True.
    store_inner_node : bool, optional
        Whether to store inner node mask in the new graph. Default: False.
    store_inner_edge : bool, optional
        Whether to store inner edge mask in the new graph. Default: False.

    Returns
    -------
    str
        The path csc_graph to save.
    """
    gpb, _, ntypes, etypes = load_partition_book(
        part_config=part_config, part_id=part_id
    )
    part = _load_part(part_config, part_id)
    part_meta = copy.deepcopy(_load_part_config(part_config))
    csc_graph = _convert_dgl_partition_to_gb(
        graph=part,
        ntypes=ntypes,
        etypes=etypes,
        gpb=gpb,
        part_meta=part_meta,
        graph_formats=graph_formats,
        store_eids=store_eids,
        store_inner_node=store_inner_node,
        store_inner_edge=store_inner_edge,
    )
    rel_path = _save_graph_gb(part_config, part_id, csc_graph)
    return rel_path


def _convert_partition_to_graphbolt_wrapper(
    graph_formats,
    part_config,
    store_eids,
    store_inner_node,
    store_inner_edge,
    n_jobs,
    num_parts,
):
    # [Rui] DGL partitions are always saved as homogeneous graphs even though
    # the original graph is heterogeneous. But heterogeneous information like
    # node/edge types are saved as node/edge data alongside with partitions.
    # What needs more attention is that due to the existence of HALO nodes in
    # each partition, the local node IDs are not sorted according to the node
    # types. So we fail to assign ``node_type_offset`` as required by GraphBolt.
    # But this is not a problem since such information is not used in sampling.
    # We can simply pass None to it.

    # Iterate over partitions.
    convert_with_format = partial(
        gb_convert_single_dgl_partition,
        part_config=part_config,
        graph_formats=graph_formats,
        store_eids=store_eids,
        store_inner_node=store_inner_node,
        store_inner_edge=store_inner_edge,
    )
    # Need to create entirely new interpreters, because we call C++ downstream
    # See https://docs.python.org/3.12/library/multiprocessing.html#contexts-and-start-methods
    # and https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil
    rel_path_results = []
    if n_jobs > 1 and num_parts > 1:
        mp_ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(  # pylint: disable=unexpected-keyword-arg
            max_workers=min(num_parts, n_jobs),
            mp_context=mp_ctx,
        ) as executor:
            for part_id in range(num_parts):
                rel_path_results.append(
                    executor.submit(
                        convert_with_format, part_id=part_id
                    ).result()
                )

    else:
        # If running single-threaded, avoid spawning new interpreter, which is slow
        for part_id in range(num_parts):
            rel_path = convert_with_format(part_id=part_id)
            rel_path_results.append(rel_path)
    part_meta = _load_part_config(part_config)
    for part_id in range(num_parts):
        # Update graph path.
        part_meta[f"part-{part_id}"]["part_graph_graphbolt"] = rel_path_results[
            part_id
        ]

    # Save dtype info into partition config.
    # [TODO][Rui] Always use int64_t for node/edge IDs in GraphBolt. See more
    # details in #7175.
    part_meta["node_map_dtype"] = "int64"
    part_meta["edge_map_dtype"] = "int64"

    return part_meta


def dgl_partition_to_graphbolt(
    part_config,
    *,
    store_eids=True,
    store_inner_node=False,
    store_inner_edge=False,
    graph_formats=None,
    n_jobs=1,
):
    """Convert partitions of dgl to FusedCSCSamplingGraph of GraphBolt.

    This API converts `DGLGraph` partitions to `FusedCSCSamplingGraph` which is
    dedicated for sampling in `GraphBolt`. New graphs will be stored alongside
    original graph as `fused_csc_sampling_graph.pt`.

    In the near future, partitions are supposed to be saved as
    `FusedCSCSamplingGraph` directly. At that time, this API should be deprecated.

    Parameters
    ----------
    part_config : str
        The partition configuration JSON file.
    store_eids : bool, optional
        Whether to store edge IDs in the new graph. Default: True.
    store_inner_node : bool, optional
        Whether to store inner node mask in the new graph. Default: False.
    store_inner_edge : bool, optional
        Whether to store inner edge mask in the new graph. Default: False.
    graph_formats : str or list[str], optional
        Save partitions in specified formats. It could be any combination of
        `coo`, `csc`. As `csc` format is mandatory for `FusedCSCSamplingGraph`,
        it is not necessary to specify this argument. It's mainly for
        specifying `coo` format to save edge ID mapping and destination node
        IDs. If not specified, whether to save `coo` format is determined by
        the availability of the format in DGL partitions. Default: None.
    n_jobs: int
        Number of parallel jobs to run during partition conversion. Max parallelism
        is determined by the partition count.
    """
    debug_mode = "DGL_DIST_DEBUG" in os.environ
    if debug_mode:
        dgl_warning(
            "Running in debug mode which means all attributes of DGL partitions"
            " will be saved to the new format."
        )
    part_meta = _load_part_config(part_config)
    num_parts = part_meta["num_parts"]
    part_meta = _convert_partition_to_graphbolt_wrapper(
        graph_formats=graph_formats,
        part_config=part_config,
        store_eids=store_eids,
        store_inner_node=store_inner_node,
        store_inner_edge=store_inner_edge,
        n_jobs=n_jobs,
        num_parts=num_parts,
    )
    _dump_part_config(part_config, part_meta)
