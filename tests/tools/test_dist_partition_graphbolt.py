import json
import os
import tempfile

import dgl
import dgl.backend as F
import dgl.graphbolt as gb

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from dgl.data.utils import load_graphs, load_tensors
from dgl.distributed.partition import (
    _etype_str_to_tuple,
    _etype_tuple_to_str,
    _get_inner_edge_mask,
    _get_inner_node_mask,
    load_partition,
    RESERVED_FIELD_DTYPE,
)

from distpartitioning import array_readwriter
from distpartitioning.utils import generate_read_list
from pytest_utils import create_chunked_dataset


def _verify_metadata_gb(gpb, g, num_parts, part_id, part_sizes):
    """
    check list:
        make sure the number of nodes and edges is correct.
        make sure the number of parts is correct.
        make sure the number of nodes and edges in each part is corrcet.
    """
    assert gpb._num_nodes() == g.num_nodes()
    assert gpb._num_edges() == g.num_edges()

    assert gpb.num_partitions() == num_parts
    gpb_meta = gpb.metadata()
    assert len(gpb_meta) == num_parts
    assert len(gpb.partid2nids(part_id)) == gpb_meta[part_id]["num_nodes"]
    assert len(gpb.partid2eids(part_id)) == gpb_meta[part_id]["num_edges"]
    part_sizes.append(
        (gpb_meta[part_id]["num_nodes"], gpb_meta[part_id]["num_edges"])
    )


def _verify_local_id_gb(part_g, part_id, gpb):
    """
    check list:
        make sure the type of local id is correct.
        make sure local id have a right order.
    """
    nid = F.boolean_mask(
        part_g.node_attributes[dgl.NID],
        part_g.node_attributes["inner_node"],
    )
    local_nid = gpb.nid2localnid(nid, part_id)
    assert F.dtype(local_nid) in (F.int64, F.int32)
    assert np.all(F.asnumpy(local_nid) == np.arange(0, len(local_nid)))
    eid = F.boolean_mask(
        part_g.edge_attributes[dgl.EID],
        part_g.edge_attributes["inner_edge"],
    )
    local_eid = gpb.eid2localeid(eid, part_id)
    assert F.dtype(local_eid) in (F.int64, F.int32)
    assert np.all(np.sort(F.asnumpy(local_eid)) == np.arange(0, len(local_eid)))
    return local_nid, local_eid


def _verify_map_gb(
    part_g,
    part_id,
    gpb,
):
    """
    check list:
        make sure the map node and its data type is correct.
    """
    # Check the node map.
    local_nodes = F.boolean_mask(
        part_g.node_attributes[dgl.NID],
        part_g.node_attributes["inner_node"],
    )
    inner_node_index = F.nonzero_1d(part_g.node_attributes["inner_node"])
    mapping_nodes = gpb.partid2nids(part_id)
    assert F.dtype(mapping_nodes) in (F.int32, F.int64)
    assert np.all(
        np.sort(F.asnumpy(local_nodes)) == np.sort(F.asnumpy(mapping_nodes))
    )
    assert np.all(
        F.asnumpy(inner_node_index) == np.arange(len(inner_node_index))
    )

    # Check the edge map.

    local_edges = F.boolean_mask(
        part_g.edge_attributes[dgl.EID],
        part_g.edge_attributes["inner_edge"],
    )
    inner_edge_index = F.nonzero_1d(part_g.edge_attributes["inner_edge"])
    mapping_edges = gpb.partid2eids(part_id)
    assert F.dtype(mapping_edges) in (F.int32, F.int64)
    assert np.all(
        np.sort(F.asnumpy(local_edges)) == np.sort(F.asnumpy(mapping_edges))
    )
    assert np.all(
        F.asnumpy(inner_edge_index) == np.arange(len(inner_edge_index))
    )
    return local_nodes, local_edges


def _verify_local_and_map_id_gb(
    part_g,
    part_id,
    gpb,
    store_inner_node,
    store_inner_edge,
    store_eids,
):
    """
    check list:
        make sure local id are correct.
        make sure mapping id are correct.
    """
    if store_inner_node and store_inner_edge and store_eids:
        _verify_local_id_gb(part_g, part_id, gpb)
        _verify_map_gb(part_g, part_id, gpb)


def _get_part_IDs(part_g):
    # These are partition-local IDs.
    num_columns = part_g.csc_indptr.diff()
    part_src_ids = part_g.indices
    part_dst_ids = torch.arange(part_g.total_num_nodes).repeat_interleave(
        num_columns
    )
    # These are reshuffled global homogeneous IDs.
    part_src_ids = F.gather_row(part_g.node_attributes[dgl.NID], part_src_ids)
    part_dst_ids = F.gather_row(part_g.node_attributes[dgl.NID], part_dst_ids)
    return part_src_ids, part_dst_ids


def _verify_node_type_ID_gb(part_g, gpb):
    """
    check list:
        make sure ntype id have correct data type
    """
    part_src_ids, part_dst_ids = _get_part_IDs(part_g)
    # These are reshuffled per-type IDs.
    src_ntype_ids, part_src_ids = gpb.map_to_per_ntype(part_src_ids)
    dst_ntype_ids, part_dst_ids = gpb.map_to_per_ntype(part_dst_ids)
    # `IdMap` is in int64 by default.
    assert src_ntype_ids.dtype == F.int64
    assert dst_ntype_ids.dtype == F.int64

    with pytest.raises(dgl.utils.internal.InconsistentDtypeException):
        gpb.map_to_per_ntype(F.tensor([0], F.int32))
    with pytest.raises(dgl.utils.internal.InconsistentDtypeException):
        gpb.map_to_per_etype(F.tensor([0], F.int32))
    return (
        part_src_ids,
        part_dst_ids,
        src_ntype_ids,
        part_src_ids,
        dst_ntype_ids,
    )


def _verify_orig_edge_IDs_gb(
    g,
    orig_nids,
    orig_eids,
    part_eids,
    part_src_ids,
    part_dst_ids,
    src_ntype=None,
    dst_ntype=None,
    etype=None,
):
    """
    check list:
        make sure orig edge id are correct after
    """
    if src_ntype is not None and dst_ntype is not None:
        orig_src_nid = orig_nids[src_ntype]
        orig_dst_nid = orig_nids[dst_ntype]
    else:
        orig_src_nid = orig_nids
        orig_dst_nid = orig_nids
    orig_src_ids = F.gather_row(orig_src_nid, part_src_ids)
    orig_dst_ids = F.gather_row(orig_dst_nid, part_dst_ids)
    if etype is not None:
        orig_eids = orig_eids[etype]
    orig_eids1 = F.gather_row(orig_eids, part_eids)
    orig_eids2 = g.edge_ids(orig_src_ids, orig_dst_ids, etype=etype)
    assert len(orig_eids1) == len(orig_eids2)
    assert np.all(F.asnumpy(orig_eids1) == F.asnumpy(orig_eids2))


def _verify_orig_IDs_gb(
    part_g,
    gpb,
    g,
    is_homo=False,
    part_src_ids=None,
    part_dst_ids=None,
    src_ntype_ids=None,
    dst_ntype_ids=None,
    orig_nids=None,
    orig_eids=None,
):
    """
    check list:
        make sure orig edge id are correct.
        make sure hetero ntype id are correct.
    """
    part_eids = part_g.edge_attributes[dgl.EID]
    if is_homo:
        _verify_orig_edge_IDs_gb(
            g, orig_nids, orig_eids, part_eids, part_src_ids, part_dst_ids
        )
        local_orig_nids = orig_nids[part_g.node_attributes[dgl.NID]]
        local_orig_eids = orig_eids[part_g.edge_attributes[dgl.EID]]
        part_g.node_attributes["feats"] = F.gather_row(
            g.ndata["feats"], local_orig_nids
        )
        part_g.edge_attributes["feats"] = F.gather_row(
            g.edata["feats"], local_orig_eids
        )
    else:
        etype_ids, part_eids = gpb.map_to_per_etype(part_eids)
        # `IdMap` is in int64 by default.
        assert etype_ids.dtype == F.int64

        # These are original per-type IDs.
        for etype_id, etype in enumerate(g.canonical_etypes):
            part_src_ids1 = F.boolean_mask(part_src_ids, etype_ids == etype_id)
            src_ntype_ids1 = F.boolean_mask(
                src_ntype_ids, etype_ids == etype_id
            )
            part_dst_ids1 = F.boolean_mask(part_dst_ids, etype_ids == etype_id)
            dst_ntype_ids1 = F.boolean_mask(
                dst_ntype_ids, etype_ids == etype_id
            )
            part_eids1 = F.boolean_mask(part_eids, etype_ids == etype_id)
            assert np.all(F.asnumpy(src_ntype_ids1 == src_ntype_ids1[0]))
            assert np.all(F.asnumpy(dst_ntype_ids1 == dst_ntype_ids1[0]))
            src_ntype = g.ntypes[F.as_scalar(src_ntype_ids1[0])]
            dst_ntype = g.ntypes[F.as_scalar(dst_ntype_ids1[0])]

            _verify_orig_edge_IDs_gb(
                g,
                orig_nids,
                orig_eids,
                part_eids1,
                part_src_ids1,
                part_dst_ids1,
                src_ntype,
                dst_ntype,
                etype,
            )


def _verify_constructed_id_gb(part_sizes, gpb):
    """
    verify the part id of each node by constructed nids.
    check list:
        make sure each node' part id and its type are corect
    """
    node_map = []
    edge_map = []
    for part_i, (num_nodes, num_edges) in enumerate(part_sizes):
        node_map.append(np.ones(num_nodes) * part_i)
        edge_map.append(np.ones(num_edges) * part_i)
    node_map = np.concatenate(node_map)
    edge_map = np.concatenate(edge_map)
    nid2pid = gpb.nid2partid(F.arange(0, len(node_map)))
    assert F.dtype(nid2pid) in (F.int32, F.int64)
    assert np.all(F.asnumpy(nid2pid) == node_map)
    eid2pid = gpb.eid2partid(F.arange(0, len(edge_map)))
    assert F.dtype(eid2pid) in (F.int32, F.int64)
    assert np.all(F.asnumpy(eid2pid) == edge_map)


def _verify_IDs_gb(
    g,
    part_g,
    part_id,
    gpb,
    part_sizes,
    orig_nids,
    orig_eids,
    store_inner_node,
    store_inner_edge,
    store_eids,
    is_homo,
):
    # verify local id and mapping id
    _verify_local_and_map_id_gb(
        part_g,
        part_id,
        gpb,
        store_inner_node,
        store_inner_edge,
        store_eids,
    )

    # Verify the mapping between the reshuffled IDs and the original IDs.
    (
        part_src_ids,
        part_dst_ids,
        src_ntype_ids,
        part_src_ids,
        dst_ntype_ids,
    ) = _verify_node_type_ID_gb(part_g, gpb)

    if store_eids:
        _verify_orig_IDs_gb(
            part_g,
            gpb,
            g,
            part_src_ids=part_src_ids,
            part_dst_ids=part_dst_ids,
            src_ntype_ids=src_ntype_ids,
            dst_ntype_ids=dst_ntype_ids,
            orig_nids=orig_nids,
            orig_eids=orig_eids,
            is_homo=is_homo,
        )
    _verify_constructed_id_gb(part_sizes, gpb)


def _collect_data_gb(
    parts,
    part_g,
    gpbs,
    gpb,
    tot_node_feats,
    node_feats,
    tot_edge_feats,
    edge_feats,
    shuffled_labels,
    shuffled_edata,
    test_ntype,
    test_etype,
):
    if test_ntype != None:
        shuffled_labels.append(node_feats[test_ntype + "/label"])
        shuffled_edata.append(
            edge_feats[_etype_tuple_to_str(test_etype) + "/count"]
        )
    else:
        shuffled_labels.append(node_feats["_N/labels"])
        shuffled_edata.append(edge_feats["_N:_E:_N/feats"])
    parts.append(part_g)
    gpbs.append(gpb)
    tot_node_feats.append(node_feats)
    tot_edge_feats.append(edge_feats)


def _verify_node_feats(g, part, gpb, orig_nids, node_feats, is_homo=False):
    for ntype in g.ntypes:
        ndata = (
            part.node_attributes
            if isinstance(part, gb.FusedCSCSamplingGraph)
            else part.ndata
        )
        ntype_id = g.get_ntype_id(ntype)
        inner_node_mask = _get_inner_node_mask(
            part,
            ntype_id,
            (gpb if isinstance(part, gb.FusedCSCSamplingGraph) else None),
        )
        inner_nids = F.boolean_mask(ndata[dgl.NID], inner_node_mask)
        ntype_ids, inner_type_nids = gpb.map_to_per_ntype(inner_nids)
        partid = gpb.nid2partid(inner_type_nids, ntype)
        if is_homo:
            assert np.all(F.asnumpy(ntype_ids) == ntype_id)
            assert np.all(F.asnumpy(partid) == gpb.partid)

        if is_homo:
            orig_id = orig_nids[inner_type_nids]
        else:
            orig_id = orig_nids[ntype][inner_type_nids]
        local_nids = gpb.nid2localnid(inner_type_nids, gpb.partid, ntype)

        for name in g.nodes[ntype].data:
            if name in [dgl.NID, "inner_node"]:
                continue
            true_feats = F.gather_row(g.nodes[ntype].data[name], orig_id)
            ndata = F.gather_row(node_feats[ntype + "/" + name], local_nids)
            assert np.all(F.asnumpy(ndata == true_feats))


def _verify_edge_feats(g, part, gpb, orig_eids, edge_feats, is_homo=False):
    for etype in g.canonical_etypes:
        edata = (
            part.edge_attributes
            if isinstance(part, gb.FusedCSCSamplingGraph)
            else part.edata
        )
        etype_id = g.get_etype_id(etype)
        inner_edge_mask = _get_inner_edge_mask(part, etype_id)
        inner_eids = F.boolean_mask(edata[dgl.EID], inner_edge_mask)
        etype_ids, inner_type_eids = gpb.map_to_per_etype(inner_eids)
        partid = gpb.eid2partid(inner_type_eids, etype)
        assert np.all(F.asnumpy(etype_ids) == etype_id)
        assert np.all(F.asnumpy(partid) == gpb.partid)

        if is_homo:
            orig_id = orig_eids[inner_type_eids]
        else:
            orig_id = orig_eids[etype][inner_type_eids]
        local_eids = gpb.eid2localeid(inner_type_eids, gpb.partid, etype)

        for name in g.edges[etype].data:
            if name in [dgl.EID, "inner_edge"]:
                continue
            true_feats = F.gather_row(g.edges[etype].data[name], orig_id)
            edata = F.gather_row(
                edge_feats[_etype_tuple_to_str(etype) + "/" + name],
                local_eids,
            )
            assert np.all(F.asnumpy(edata == true_feats))


def _verify_shuffled_labels_gb(
    g,
    shuffled_labels,
    shuffled_edata,
    orig_nids,
    orig_eids,
    test_ntype=None,
    test_etype=None,
):
    """
    check list:
        make sure node data are correct.
        make sure edge data are correct.
    """
    shuffled_labels = F.asnumpy(F.cat(shuffled_labels, 0))
    shuffled_edata = F.asnumpy(F.cat(shuffled_edata, 0))
    orig_labels = np.zeros(shuffled_labels.shape, dtype=shuffled_labels.dtype)
    orig_edata = np.zeros(shuffled_edata.shape, dtype=shuffled_edata.dtype)

    orig_nid = orig_nids if test_ntype is None else orig_nids[test_ntype]
    orig_eid = orig_eids if test_etype is None else orig_eids[test_etype]
    nlabel = (
        g.ndata["labels"]
        if test_ntype is None
        else g.nodes[test_ntype].data["label"]
    )
    edata = (
        g.edata["feats"]
        if test_etype is None
        else g.edges[test_etype].data["count"]
    )

    orig_labels[F.asnumpy(orig_nid)] = shuffled_labels
    orig_edata[F.asnumpy(orig_eid)] = shuffled_edata
    assert np.all(orig_labels == F.asnumpy(nlabel))
    assert np.all(orig_edata == F.asnumpy(edata))


def verify_graph_feats_gb(
    g,
    gpbs,
    parts,
    tot_node_feats,
    tot_edge_feats,
    orig_nids,
    orig_eids,
    shuffled_labels,
    shuffled_edata,
    test_ntype,
    test_etype,
    store_inner_node=False,
    store_inner_edge=False,
    store_eids=False,
    is_homo=False,
):
    """
    check list:
        make sure the feats of nodes and edges are correct
    """
    for part_id in range(len(parts)):
        part = parts[part_id]
        gpb = gpbs[part_id]
        node_feats = tot_node_feats[part_id]
        edge_feats = tot_edge_feats[part_id]
        if store_inner_node:
            _verify_node_feats(
                g,
                part,
                gpb,
                orig_nids,
                node_feats,
                is_homo=is_homo,
            )
        if store_inner_edge and store_eids:
            _verify_edge_feats(
                g,
                part,
                gpb,
                orig_eids,
                edge_feats,
                is_homo=is_homo,
            )

    _verify_shuffled_labels_gb(
        g,
        shuffled_labels,
        shuffled_edata,
        orig_nids,
        orig_eids,
        test_ntype,
        test_etype,
    )


def _verify_graphbolt_attributes(
    parts, store_inner_node, store_inner_edge, store_eids
):
    """
    check list:
        make sure arguments work.
    """
    for part in parts:
        assert store_inner_edge == ("inner_edge" in part.edge_attributes)
        assert store_inner_node == ("inner_node" in part.node_attributes)
        assert store_eids == (dgl.EID in part.edge_attributes)


def _verify_graphbolt_part(
    g,
    test_dir,
    orig_nids,
    orig_eids,
    graph_name,
    num_parts,
    store_inner_node,
    store_inner_edge,
    store_eids,
    part_config=None,
    test_ntype=None,
    test_etype=None,
    is_homo=False,
):
    """
    check list:
        _verify_metadata_gb:
            data type, ID's order and ID's number of edges and nodes
        _verify_IDs_gb:
            local id, mapping id,node type id, orig edge, hetero ntype id
        verify_graph_feats_gb:
            nodes and edges' feats
        _verify_graphbolt_attributes:
            arguments
    """
    parts = []
    tot_node_feats = []
    tot_edge_feats = []
    shuffled_labels = []
    shuffled_edata = []
    part_sizes = []
    gpbs = []
    if part_config is None:
        part_config = os.path.join(test_dir, f"{graph_name}.json")
    # test each part
    for part_id in range(num_parts):
        part_g, node_feats, edge_feats, gpb, _, _, _ = load_partition(
            part_config, part_id, load_feats=True, use_graphbolt=True
        )
        # verify metadata
        _verify_metadata_gb(
            gpb,
            g,
            num_parts,
            part_id,
            part_sizes,
        )

        # verify eid and nid
        _verify_IDs_gb(
            g,
            part_g,
            part_id,
            gpb,
            part_sizes,
            orig_nids,
            orig_eids,
            store_inner_node,
            store_inner_edge,
            store_eids,
            is_homo,
        )

        # collect shuffled data and parts
        _collect_data_gb(
            parts,
            part_g,
            gpbs,
            gpb,
            tot_node_feats,
            node_feats,
            tot_edge_feats,
            edge_feats,
            shuffled_labels,
            shuffled_edata,
            test_ntype,
            test_etype,
        )

    # verify graph feats
    verify_graph_feats_gb(
        g,
        gpbs,
        parts,
        tot_node_feats,
        tot_edge_feats,
        orig_nids,
        orig_eids,
        shuffled_labels=shuffled_labels,
        shuffled_edata=shuffled_edata,
        test_ntype=test_ntype,
        test_etype=test_etype,
        store_inner_node=store_inner_node,
        store_inner_edge=store_inner_edge,
        store_eids=store_eids,
        is_homo=is_homo,
    )

    _verify_graphbolt_attributes(
        parts, store_inner_node, store_inner_edge, store_eids
    )

    return parts


def _verify_hetero_graph_node_edge_num(
    g,
    parts,
    store_inner_edge,
    debug_mode,
):
    """
    check list:
        make sure edge type are correct.
        make sure the number of nodes in each node type are correct.
        make sure the number of nodes in each node type are correct.
    """
    num_nodes = {ntype: 0 for ntype in g.ntypes}
    num_edges = {etype: 0 for etype in g.canonical_etypes}
    for part in parts:
        edata = (
            part.edge_attributes
            if isinstance(part, gb.FusedCSCSamplingGraph)
            else part.edata
        )
        if dgl.ETYPE in edata:
            assert len(g.canonical_etypes) == len(F.unique(edata[dgl.ETYPE]))
        if debug_mode or isinstance(part, dgl.DGLGraph):
            for ntype in g.ntypes:
                ntype_id = g.get_ntype_id(ntype)
                inner_node_mask = _get_inner_node_mask(part, ntype_id)
                num_inner_nodes = F.sum(F.astype(inner_node_mask, F.int64), 0)
                num_nodes[ntype] += num_inner_nodes
        if store_inner_edge or isinstance(part, dgl.DGLGraph):
            for etype in g.canonical_etypes:
                etype_id = g.get_etype_id(etype)
                inner_edge_mask = _get_inner_edge_mask(part, etype_id)
                num_inner_edges = F.sum(F.astype(inner_edge_mask, F.int64), 0)
                num_edges[etype] += num_inner_edges

    # Verify the number of nodes are correct.
    if debug_mode or isinstance(part, dgl.DGLGraph):
        for ntype in g.ntypes:
            print(
                "node {}: {}, {}".format(
                    ntype, g.num_nodes(ntype), num_nodes[ntype]
                )
            )
            assert g.num_nodes(ntype) == num_nodes[ntype]
    # Verify the number of edges are correct.
    if store_inner_edge or isinstance(part, dgl.DGLGraph):
        for etype in g.canonical_etypes:
            print(
                "edge {}: {}, {}".format(
                    etype, g.num_edges(etype), num_edges[etype]
                )
            )
            assert g.num_edges(etype) == num_edges[etype]


def _verify_edge_id_range_hetero(
    g,
    part,
    eids,
):
    """
    check list:
        make sure inner_eids fall into a range.
        make sure all edges are included.
    """
    edata = (
        part.edge_attributes
        if isinstance(part, gb.FusedCSCSamplingGraph)
        else part.edata
    )
    etype = (
        part.type_per_edge
        if isinstance(part, gb.FusedCSCSamplingGraph)
        else edata[dgl.ETYPE]
    )
    eid = torch.arange(len(edata[dgl.EID]))
    etype_arr = F.gather_row(etype, eid)
    eid_arr = F.gather_row(edata[dgl.EID], eid)
    for etype in g.canonical_etypes:
        etype_id = g.get_etype_id(etype)
        eids[etype].append(F.boolean_mask(eid_arr, etype_arr == etype_id))
        # Make sure edge Ids fall into a range.
        inner_edge_mask = _get_inner_edge_mask(part, etype_id)
        inner_eids = np.sort(
            F.asnumpy(F.boolean_mask(edata[dgl.EID], inner_edge_mask))
        )
        assert np.all(
            inner_eids == np.arange(inner_eids[0], inner_eids[-1] + 1)
        )
    return eids


def _verify_node_id_range_hetero(g, part, nids):
    """
    check list:
        make sure inner nodes have Ids fall into a range.
    """
    for ntype in g.ntypes:
        ntype_id = g.get_ntype_id(ntype)
        # Make sure inner nodes have Ids fall into a range.
        inner_node_mask = _get_inner_node_mask(part, ntype_id)
        inner_nids = F.boolean_mask(
            part.node_attributes[dgl.NID], inner_node_mask
        )
        assert np.all(
            F.asnumpy(
                inner_nids
                == F.arange(
                    F.as_scalar(inner_nids[0]),
                    F.as_scalar(inner_nids[-1]) + 1,
                )
            )
        )
        nids[ntype].append(inner_nids)
    return nids


def _verify_graph_attributes_hetero(
    g,
    parts,
    store_inner_edge,
    store_inner_node,
):
    """
    check list:
        make sure edge ids fall into a range.
        make sure inner nodes have Ids fall into a range.
        make sure all nodes is included.
        make sure all edges is included.
    """
    nids = {ntype: [] for ntype in g.ntypes}
    eids = {etype: [] for etype in g.canonical_etypes}
    # check edge id.
    if store_inner_edge or isinstance(parts[0], dgl.DGLGraph):
        for part in parts:
            # collect eids
            eids = _verify_edge_id_range_hetero(g, part, eids)
        for etype in eids:
            eids_type = F.cat(eids[etype], 0)
            uniq_ids = F.unique(eids_type)
            # We should get all nodes.
            assert len(uniq_ids) == g.num_edges(etype)

    # check node id.
    if store_inner_node or isinstance(parts[0], dgl.DGLGraph):
        for part in parts:
            nids = _verify_node_id_range_hetero(g, part, nids)
        for ntype in nids:
            nids_type = F.cat(nids[ntype], 0)
            uniq_ids = F.unique(nids_type)
            # We should get all nodes.
            assert len(uniq_ids) == g.num_nodes(ntype)


def _verify_hetero_graph(
    g,
    parts,
    store_eids=False,
    store_inner_edge=False,
    store_inner_node=False,
    debug_mode=False,
):
    _verify_hetero_graph_node_edge_num(
        g,
        parts,
        store_inner_edge=store_inner_edge,
        debug_mode=debug_mode,
    )
    if store_eids:
        _verify_graph_attributes_hetero(
            g,
            parts,
            store_inner_edge=store_inner_edge,
            store_inner_node=store_inner_node,
        )


def _test_pipeline_graphbolt(
    num_chunks,
    num_parts,
    world_size,
    graph_formats=None,
    data_fmt="numpy",
    num_chunks_nodes=None,
    num_chunks_edges=None,
    num_chunks_node_data=None,
    num_chunks_edge_data=None,
    use_verify_partitions=False,
    store_eids=True,
    store_inner_edge=True,
    store_inner_node=True,
):
    if num_parts % world_size != 0:
        # num_parts should be a multiple of world_size
        return

    with tempfile.TemporaryDirectory() as root_dir:
        g = create_chunked_dataset(
            root_dir,
            num_chunks,
            data_fmt=data_fmt,
            num_chunks_nodes=num_chunks_nodes,
            num_chunks_edges=num_chunks_edges,
            num_chunks_node_data=num_chunks_node_data,
            num_chunks_edge_data=num_chunks_edge_data,
        )
        graph_name = "test"
        test_ntype = "paper"
        test_etype = ("paper", "cites", "paper")

        # Step1: graph partition
        in_dir = os.path.join(root_dir, "chunked-data")
        output_dir = os.path.join(root_dir, "parted_data")
        os.system(
            "python3 tools/partition_algo/random_partition.py "
            "--in_dir {} --out_dir {} --num_partitions {}".format(
                in_dir, output_dir, num_parts
            )
        )
        for ntype in ["author", "institution", "paper"]:
            fname = os.path.join(output_dir, "{}.txt".format(ntype))
            with open(fname, "r") as f:
                header = f.readline().rstrip()
                assert isinstance(int(header), int)

        # Step2: data dispatch
        partition_dir = os.path.join(root_dir, "parted_data")
        out_dir = os.path.join(root_dir, "partitioned")
        ip_config = os.path.join(root_dir, "ip_config.txt")
        with open(ip_config, "w") as f:
            for i in range(world_size):
                f.write(f"127.0.0.{i + 1}\n")

        cmd = "python3 tools/dispatch_data.py "
        cmd += f" --in-dir {in_dir} "
        cmd += f" --partitions-dir {partition_dir} "
        cmd += f" --out-dir {out_dir} "
        cmd += f" --ip-config {ip_config} "
        cmd += " --ssh-port 22 "
        cmd += " --process-group-timeout 60 "
        cmd += " --save-orig-nids "
        cmd += " --save-orig-eids "
        cmd += " --use-graphbolt "
        cmd += f" --graph-formats {graph_formats} " if graph_formats else ""

        if store_eids:
            cmd += " --store-eids "
        if store_inner_edge:
            cmd += " --store-inner-edge "
        if store_inner_node:
            cmd += " --store-inner-node "
        os.system(cmd)

        # check if verify_partitions.py is used for validation.
        if use_verify_partitions:
            cmd = "python3 tools/verify_partitions.py "
            cmd += f" --orig-dataset-dir {in_dir}"
            cmd += f" --part-graph {out_dir}"
            cmd += f" --partitions-dir {output_dir}"
            os.system(cmd)
            return

        # read original node/edge IDs
        def read_orig_ids(fname):
            orig_ids = {}
            for i in range(num_parts):
                ids_path = os.path.join(out_dir, f"part{i}", fname)
                part_ids = load_tensors(ids_path)
                for type, data in part_ids.items():
                    if type not in orig_ids:
                        orig_ids[type] = data
                    else:
                        orig_ids[type] = torch.cat((orig_ids[type], data))
            return orig_ids

        orig_nids, orig_eids = None, None
        orig_nids = read_orig_ids("orig_nids.dgl")

        orig_eids_str = read_orig_ids("orig_eids.dgl")

        orig_eids = {}
        # transmit etype from string to tuple.
        for etype, eids in orig_eids_str.items():
            orig_eids[_etype_str_to_tuple(etype)] = eids

        # load partitions and verify
        part_config = os.path.join(out_dir, "metadata.json")
        parts = _verify_graphbolt_part(
            g,
            root_dir,
            orig_nids,
            orig_eids,
            graph_name,
            num_parts,
            store_inner_node,
            store_inner_edge,
            store_eids,
            test_ntype=test_ntype,
            test_etype=test_etype,
            part_config=part_config,
            is_homo=False,
        )
        _verify_hetero_graph(
            g,
            parts,
            store_eids=store_eids,
            store_inner_edge=store_inner_edge,
        )


@pytest.mark.parametrize(
    "num_chunks, num_parts, world_size",
    [[4, 4, 4], [8, 4, 2], [8, 4, 4], [9, 6, 3], [11, 11, 1], [11, 4, 1]],
)
def test_pipeline_basics(num_chunks, num_parts, world_size):
    _test_pipeline_graphbolt(
        num_chunks,
        num_parts,
        world_size,
    )
    _test_pipeline_graphbolt(
        num_chunks, num_parts, world_size, use_verify_partitions=False
    )


@pytest.mark.parametrize("store_inner_node", [True, False])
@pytest.mark.parametrize("store_inner_edge", [True, False])
@pytest.mark.parametrize("store_eids", [True, False])
def test_pipeline_attributes(store_inner_node, store_inner_edge, store_eids):
    _test_pipeline_graphbolt(
        4,
        4,
        4,
        store_inner_node=store_inner_node,
        store_inner_edge=store_inner_edge,
        store_eids=store_eids,
    )


@pytest.mark.parametrize(
    "num_chunks, "
    "num_parts, "
    "world_size, "
    "num_chunks_node_data, "
    "num_chunks_edge_data",
    [
        # Test cases where no. of chunks more than
        # no. of partitions
        [8, 4, 4, 8, 8],
        [8, 4, 2, 8, 8],
        [9, 7, 5, 9, 9],
        [8, 8, 4, 8, 8],
        # Test cases where no. of chunks smaller
        # than no. of partitions
        [7, 8, 4, 7, 7],
        [1, 8, 4, 1, 1],
        [1, 4, 4, 1, 1],
        [3, 4, 4, 3, 3],
        [1, 4, 2, 1, 1],
        [3, 4, 2, 3, 3],
        [1, 5, 3, 1, 1],
    ],
)
def test_pipeline_arbitrary_chunks(
    num_chunks,
    num_parts,
    world_size,
    num_chunks_node_data,
    num_chunks_edge_data,
):

    _test_pipeline_graphbolt(
        num_chunks,
        num_parts,
        world_size,
        num_chunks_node_data=num_chunks_node_data,
        num_chunks_edge_data=num_chunks_edge_data,
    )


@pytest.mark.parametrize("data_fmt", ["numpy", "parquet"])
def test_pipeline_feature_format(data_fmt):
    _test_pipeline_graphbolt(4, 4, 4, data_fmt=data_fmt)
