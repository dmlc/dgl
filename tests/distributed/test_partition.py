import json
import os
import tempfile

import dgl

import dgl.backend as F
import dgl.graphbolt as gb
import numpy as np
import pytest
import torch as th
from dgl import function as fn
from dgl.distributed import (
    dgl_partition_to_graphbolt,
    load_partition,
    load_partition_book,
    load_partition_feats,
    partition_graph,
)
from dgl.distributed.graph_partition_book import (
    _etype_str_to_tuple,
    _etype_tuple_to_str,
    DEFAULT_ETYPE,
    DEFAULT_NTYPE,
    EdgePartitionPolicy,
    HeteroDataName,
    NodePartitionPolicy,
    RangePartitionBook,
)
from dgl.distributed.partition import (
    _get_inner_edge_mask,
    _get_inner_node_mask,
    RESERVED_FIELD_DTYPE,
)
from scipy import sparse as spsp
from utils import reset_envs


def _verify_partition_data_types(part_g):
    """
    check list:
        make sure nodes and edges have correct type.
    """
    ndata = (
        part_g.node_attributes
        if isinstance(part_g, gb.FusedCSCSamplingGraph)
        else part_g.ndata
    )
    edata = (
        part_g.edge_attributes
        if isinstance(part_g, gb.FusedCSCSamplingGraph)
        else part_g.edata
    )

    for k, dtype in RESERVED_FIELD_DTYPE.items():
        if k in ndata:
            assert ndata[k].dtype == dtype
        if k in edata:
            assert edata[k].dtype == dtype


def _verify_partition_formats(part_g, formats):
    # verify saved graph formats
    if formats is None:
        assert "coo" in part_g.formats()["created"]
    else:
        for format in formats:
            assert format in part_g.formats()["created"]


def create_random_graph(n):
    arr = (
        spsp.random(n, n, density=0.001, format="coo", random_state=100) != 0
    ).astype(np.int64)
    return dgl.from_scipy(arr)


def create_random_hetero():
    num_nodes = {"n1": 1000, "n2": 1010, "n3": 1020}
    etypes = [
        ("n1", "r1", "n2"),
        ("n2", "r1", "n1"),
        ("n1", "r2", "n3"),
        ("n2", "r3", "n3"),
    ]
    edges = {}
    for etype in etypes:
        src_ntype, _, dst_ntype = etype
        arr = spsp.random(
            num_nodes[src_ntype],
            num_nodes[dst_ntype],
            density=0.001,
            format="coo",
            random_state=100,
        )
        edges[etype] = (arr.row, arr.col)
    return dgl.heterograph(edges, num_nodes)


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
            # edata may not contain all edge types.
            assert len(g.canonical_etypes) >= len(F.unique(edata[dgl.ETYPE]))
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
    eid = th.arange(len(edata[dgl.EID]))
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


def verify_graph_feats_hetero_dgl(
    g,
    gpb,
    part,
    node_feats,
    edge_feats,
    orig_nids,
    orig_eids,
):
    """
    check list:
        make sure the feats of nodes and edges are correct
    """
    _verify_node_feats(g, part, gpb, orig_nids, node_feats)

    _verify_edge_feats(g, part, gpb, orig_eids, edge_feats)


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


def check_hetero_partition(
    hg,
    part_method,
    num_parts=4,
    num_trainers_per_machine=1,
    load_feats=True,
    graph_formats=None,
):
    test_ntype = "n1"
    test_etype = ("n1", "r1", "n2")
    hg.nodes[test_ntype].data["labels"] = F.arange(0, hg.num_nodes(test_ntype))
    hg.nodes[test_ntype].data["feats"] = F.tensor(
        np.random.randn(hg.num_nodes(test_ntype), 10), F.float32
    )
    hg.edges[test_etype].data["feats"] = F.tensor(
        np.random.randn(hg.num_edges(test_etype), 10), F.float32
    )
    hg.edges[test_etype].data["labels"] = F.arange(0, hg.num_edges(test_etype))
    num_hops = 1

    orig_nids, orig_eids = partition_graph(
        hg,
        "test",
        num_parts,
        "/tmp/partition",
        num_hops=num_hops,
        part_method=part_method,
        return_mapping=True,
        num_trainers_per_machine=num_trainers_per_machine,
        graph_formats=graph_formats,
    )
    assert len(orig_nids) == len(hg.ntypes)
    assert len(orig_eids) == len(hg.canonical_etypes)
    for ntype in hg.ntypes:
        assert len(orig_nids[ntype]) == hg.num_nodes(ntype)
    for etype in hg.canonical_etypes:
        assert len(orig_eids[etype]) == hg.num_edges(etype)
    parts = []
    shuffled_labels = []
    shuffled_elabels = []
    for i in range(num_parts):
        part_g, node_feats, edge_feats, gpb, _, _, _ = load_partition(
            "/tmp/partition/test.json", i, load_feats=load_feats
        )
        _verify_partition_data_types(part_g)
        _verify_partition_formats(part_g, graph_formats)
        if not load_feats:
            assert not node_feats
            assert not edge_feats
            node_feats, edge_feats = load_partition_feats(
                "/tmp/partition/test.json", i
            )
        if num_trainers_per_machine > 1:
            for ntype in hg.ntypes:
                name = ntype + "/trainer_id"
                assert name in node_feats
                part_ids = F.floor_div(
                    node_feats[name], num_trainers_per_machine
                )
                assert np.all(F.asnumpy(part_ids) == i)

            for etype in hg.canonical_etypes:
                name = _etype_tuple_to_str(etype) + "/trainer_id"
                assert name in edge_feats
                part_ids = F.floor_div(
                    edge_feats[name], num_trainers_per_machine
                )
                assert np.all(F.asnumpy(part_ids) == i)
        # Verify the mapping between the reshuffled IDs and the original IDs.
        # These are partition-local IDs.
        part_src_ids, part_dst_ids = part_g.edges()
        # These are reshuffled global homogeneous IDs.
        part_src_ids = F.gather_row(part_g.ndata[dgl.NID], part_src_ids)
        part_dst_ids = F.gather_row(part_g.ndata[dgl.NID], part_dst_ids)
        part_eids = part_g.edata[dgl.EID]
        # These are reshuffled per-type IDs.
        src_ntype_ids, part_src_ids = gpb.map_to_per_ntype(part_src_ids)
        dst_ntype_ids, part_dst_ids = gpb.map_to_per_ntype(part_dst_ids)
        etype_ids, part_eids = gpb.map_to_per_etype(part_eids)
        # `IdMap` is in int64 by default.
        assert src_ntype_ids.dtype == F.int64
        assert dst_ntype_ids.dtype == F.int64
        assert etype_ids.dtype == F.int64
        with pytest.raises(dgl.utils.internal.InconsistentDtypeException):
            gpb.map_to_per_ntype(F.tensor([0], F.int32))
        with pytest.raises(dgl.utils.internal.InconsistentDtypeException):
            gpb.map_to_per_etype(F.tensor([0], F.int32))
        # These are original per-type IDs.
        for etype_id, etype in enumerate(hg.canonical_etypes):
            if F.sum((etype_ids == etype_id), 0) == 0:
                continue
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
            src_ntype = hg.ntypes[F.as_scalar(src_ntype_ids1[0])]
            dst_ntype = hg.ntypes[F.as_scalar(dst_ntype_ids1[0])]
            orig_src_ids1 = F.gather_row(orig_nids[src_ntype], part_src_ids1)
            orig_dst_ids1 = F.gather_row(orig_nids[dst_ntype], part_dst_ids1)
            orig_eids1 = F.gather_row(orig_eids[etype], part_eids1)
            orig_eids2 = hg.edge_ids(orig_src_ids1, orig_dst_ids1, etype=etype)
            assert len(orig_eids1) == len(orig_eids2)
            assert np.all(F.asnumpy(orig_eids1) == F.asnumpy(orig_eids2))
        parts.append(part_g)
        verify_graph_feats_hetero_dgl(
            hg, gpb, part_g, node_feats, edge_feats, orig_nids, orig_eids
        )

        shuffled_labels.append(node_feats[test_ntype + "/labels"])
        shuffled_elabels.append(
            edge_feats[_etype_tuple_to_str(test_etype) + "/labels"]
        )
    _verify_hetero_graph(hg, parts)
    shuffled_labels = F.asnumpy(F.cat(shuffled_labels, 0))
    shuffled_elabels = F.asnumpy(F.cat(shuffled_elabels, 0))
    orig_labels = np.zeros(shuffled_labels.shape, dtype=shuffled_labels.dtype)
    orig_elabels = np.zeros(
        shuffled_elabels.shape, dtype=shuffled_elabels.dtype
    )
    orig_labels[F.asnumpy(orig_nids[test_ntype])] = shuffled_labels
    orig_elabels[F.asnumpy(orig_eids[test_etype])] = shuffled_elabels
    assert np.all(orig_labels == F.asnumpy(hg.nodes[test_ntype].data["labels"]))
    assert np.all(
        orig_elabels == F.asnumpy(hg.edges[test_etype].data["labels"])
    )


def check_partition(
    g,
    part_method,
    num_parts=4,
    num_trainers_per_machine=1,
    load_feats=True,
    graph_formats=None,
):
    g.ndata["labels"] = F.arange(0, g.num_nodes())
    g.ndata["feats"] = F.tensor(np.random.randn(g.num_nodes(), 10), F.float32)
    g.edata["feats"] = F.tensor(np.random.randn(g.num_edges(), 10), F.float32)
    g.update_all(fn.copy_u("feats", "msg"), fn.sum("msg", "h"))
    g.update_all(fn.copy_e("feats", "msg"), fn.sum("msg", "eh"))
    num_hops = 2

    orig_nids, orig_eids = partition_graph(
        g,
        "test",
        num_parts,
        "/tmp/partition",
        num_hops=num_hops,
        part_method=part_method,
        return_mapping=True,
        num_trainers_per_machine=num_trainers_per_machine,
        graph_formats=graph_formats,
    )
    part_sizes = []
    shuffled_labels = []
    shuffled_edata = []
    for i in range(num_parts):
        part_g, node_feats, edge_feats, gpb, _, _, _ = load_partition(
            "/tmp/partition/test.json", i, load_feats=load_feats
        )
        _verify_partition_data_types(part_g)
        _verify_partition_formats(part_g, graph_formats)
        if not load_feats:
            assert not node_feats
            assert not edge_feats
            node_feats, edge_feats = load_partition_feats(
                "/tmp/partition/test.json", i
            )
        if num_trainers_per_machine > 1:
            for ntype in g.ntypes:
                name = ntype + "/trainer_id"
                assert name in node_feats
                part_ids = F.floor_div(
                    node_feats[name], num_trainers_per_machine
                )
                assert np.all(F.asnumpy(part_ids) == i)

            for etype in g.canonical_etypes:
                name = _etype_tuple_to_str(etype) + "/trainer_id"
                assert name in edge_feats
                part_ids = F.floor_div(
                    edge_feats[name], num_trainers_per_machine
                )
                assert np.all(F.asnumpy(part_ids) == i)

        # Check the metadata
        assert gpb._num_nodes() == g.num_nodes()
        assert gpb._num_edges() == g.num_edges()

        assert gpb.num_partitions() == num_parts
        gpb_meta = gpb.metadata()
        assert len(gpb_meta) == num_parts
        assert len(gpb.partid2nids(i)) == gpb_meta[i]["num_nodes"]
        assert len(gpb.partid2eids(i)) == gpb_meta[i]["num_edges"]
        part_sizes.append((gpb_meta[i]["num_nodes"], gpb_meta[i]["num_edges"]))

        nid = F.boolean_mask(part_g.ndata[dgl.NID], part_g.ndata["inner_node"])
        local_nid = gpb.nid2localnid(nid, i)
        assert F.dtype(local_nid) in (F.int64, F.int32)
        assert np.all(F.asnumpy(local_nid) == np.arange(0, len(local_nid)))
        eid = F.boolean_mask(part_g.edata[dgl.EID], part_g.edata["inner_edge"])
        local_eid = gpb.eid2localeid(eid, i)
        assert F.dtype(local_eid) in (F.int64, F.int32)
        assert np.all(F.asnumpy(local_eid) == np.arange(0, len(local_eid)))

        # Check the node map.
        local_nodes = F.boolean_mask(
            part_g.ndata[dgl.NID], part_g.ndata["inner_node"]
        )
        llocal_nodes = F.nonzero_1d(part_g.ndata["inner_node"])
        local_nodes1 = gpb.partid2nids(i)
        assert F.dtype(local_nodes1) in (F.int32, F.int64)
        assert np.all(
            np.sort(F.asnumpy(local_nodes)) == np.sort(F.asnumpy(local_nodes1))
        )
        assert np.all(F.asnumpy(llocal_nodes) == np.arange(len(llocal_nodes)))

        # Check the edge map.
        local_edges = F.boolean_mask(
            part_g.edata[dgl.EID], part_g.edata["inner_edge"]
        )
        llocal_edges = F.nonzero_1d(part_g.edata["inner_edge"])
        local_edges1 = gpb.partid2eids(i)
        assert F.dtype(local_edges1) in (F.int32, F.int64)
        assert np.all(
            np.sort(F.asnumpy(local_edges)) == np.sort(F.asnumpy(local_edges1))
        )
        assert np.all(F.asnumpy(llocal_edges) == np.arange(len(llocal_edges)))

        # Verify the mapping between the reshuffled IDs and the original IDs.
        part_src_ids, part_dst_ids = part_g.edges()
        part_src_ids = F.gather_row(part_g.ndata[dgl.NID], part_src_ids)
        part_dst_ids = F.gather_row(part_g.ndata[dgl.NID], part_dst_ids)
        part_eids = part_g.edata[dgl.EID]
        orig_src_ids = F.gather_row(orig_nids, part_src_ids)
        orig_dst_ids = F.gather_row(orig_nids, part_dst_ids)
        orig_eids1 = F.gather_row(orig_eids, part_eids)
        orig_eids2 = g.edge_ids(orig_src_ids, orig_dst_ids)
        assert F.shape(orig_eids1)[0] == F.shape(orig_eids2)[0]
        assert np.all(F.asnumpy(orig_eids1) == F.asnumpy(orig_eids2))

        local_orig_nids = orig_nids[part_g.ndata[dgl.NID]]
        local_orig_eids = orig_eids[part_g.edata[dgl.EID]]
        part_g.ndata["feats"] = F.gather_row(g.ndata["feats"], local_orig_nids)
        part_g.edata["feats"] = F.gather_row(g.edata["feats"], local_orig_eids)
        local_nodes = orig_nids[local_nodes]
        local_edges = orig_eids[local_edges]

        part_g.update_all(fn.copy_u("feats", "msg"), fn.sum("msg", "h"))
        part_g.update_all(fn.copy_e("feats", "msg"), fn.sum("msg", "eh"))
        assert F.allclose(
            F.gather_row(g.ndata["h"], local_nodes),
            F.gather_row(part_g.ndata["h"], llocal_nodes),
        )
        assert F.allclose(
            F.gather_row(g.ndata["eh"], local_nodes),
            F.gather_row(part_g.ndata["eh"], llocal_nodes),
        )

        for name in ["labels", "feats"]:
            assert "_N/" + name in node_feats
            assert node_feats["_N/" + name].shape[0] == len(local_nodes)
            true_feats = F.gather_row(g.ndata[name], local_nodes)
            ndata = F.gather_row(node_feats["_N/" + name], local_nid)
            assert np.all(F.asnumpy(true_feats) == F.asnumpy(ndata))
        for name in ["feats"]:
            efeat_name = _etype_tuple_to_str(DEFAULT_ETYPE) + "/" + name
            assert efeat_name in edge_feats
            assert edge_feats[efeat_name].shape[0] == len(local_edges)
            true_feats = F.gather_row(g.edata[name], local_edges)
            edata = F.gather_row(edge_feats[efeat_name], local_eid)
            assert np.all(F.asnumpy(true_feats) == F.asnumpy(edata))

        # This only works if node/edge IDs are shuffled.
        shuffled_labels.append(node_feats["_N/labels"])
        shuffled_edata.append(edge_feats["_N:_E:_N/feats"])

    # Verify that we can reconstruct node/edge data for original IDs.
    shuffled_labels = F.asnumpy(F.cat(shuffled_labels, 0))
    shuffled_edata = F.asnumpy(F.cat(shuffled_edata, 0))
    orig_labels = np.zeros(shuffled_labels.shape, dtype=shuffled_labels.dtype)
    orig_edata = np.zeros(shuffled_edata.shape, dtype=shuffled_edata.dtype)
    orig_labels[F.asnumpy(orig_nids)] = shuffled_labels
    orig_edata[F.asnumpy(orig_eids)] = shuffled_edata
    assert np.all(orig_labels == F.asnumpy(g.ndata["labels"]))
    assert np.all(orig_edata == F.asnumpy(g.edata["feats"]))

    node_map = []
    edge_map = []
    for i, (num_nodes, num_edges) in enumerate(part_sizes):
        node_map.append(np.ones(num_nodes) * i)
        edge_map.append(np.ones(num_edges) * i)
    node_map = np.concatenate(node_map)
    edge_map = np.concatenate(edge_map)
    nid2pid = gpb.nid2partid(F.arange(0, len(node_map)))
    assert F.dtype(nid2pid) in (F.int32, F.int64)
    assert np.all(F.asnumpy(nid2pid) == node_map)
    eid2pid = gpb.eid2partid(F.arange(0, len(edge_map)))
    assert F.dtype(eid2pid) in (F.int32, F.int64)
    assert np.all(F.asnumpy(eid2pid) == edge_map)


@pytest.mark.parametrize("part_method", ["metis", "random"])
@pytest.mark.parametrize("num_parts", [1, 4])
@pytest.mark.parametrize("num_trainers_per_machine", [1])
@pytest.mark.parametrize("load_feats", [True, False])
@pytest.mark.parametrize(
    "graph_formats", [None, ["csc"], ["coo", "csc"], ["coo", "csc", "csr"]]
)
def test_partition(
    part_method,
    num_parts,
    num_trainers_per_machine,
    load_feats,
    graph_formats,
):
    os.environ["DGL_DIST_DEBUG"] = "1"
    if part_method == "random" and num_parts > 1:
        num_trainers_per_machine = 1
    g = create_random_graph(1000)
    check_partition(
        g,
        part_method,
        num_parts,
        num_trainers_per_machine,
        load_feats,
        graph_formats,
    )
    hg = create_random_hetero()
    check_hetero_partition(
        hg,
        part_method,
        num_parts,
        num_trainers_per_machine,
        load_feats,
        graph_formats,
    )
    reset_envs()


@pytest.mark.parametrize("node_map_dtype", [F.int32, F.int64])
@pytest.mark.parametrize("edge_map_dtype", [F.int32, F.int64])
def test_RangePartitionBook(node_map_dtype, edge_map_dtype):
    part_id = 1
    num_parts = 2

    # homogeneous
    node_map = {
        DEFAULT_NTYPE: F.tensor([[0, 1000], [1000, 2000]], dtype=node_map_dtype)
    }
    edge_map = {
        DEFAULT_ETYPE: F.tensor(
            [[0, 5000], [5000, 10000]], dtype=edge_map_dtype
        )
    }
    ntypes = {DEFAULT_NTYPE: 0}
    etypes = {DEFAULT_ETYPE: 0}
    gpb = RangePartitionBook(
        part_id, num_parts, node_map, edge_map, ntypes, etypes
    )
    assert gpb.etypes == [DEFAULT_ETYPE[1]]
    assert gpb.canonical_etypes == [DEFAULT_ETYPE]
    assert gpb.to_canonical_etype(DEFAULT_ETYPE[1]) == DEFAULT_ETYPE
    ntype_ids, per_ntype_ids = gpb.map_to_per_ntype(
        F.tensor([0, 1000], dtype=node_map_dtype)
    )
    assert ntype_ids.dtype == node_map_dtype
    assert per_ntype_ids.dtype == node_map_dtype
    assert np.all(F.asnumpy(ntype_ids) == 0)
    assert np.all(F.asnumpy(per_ntype_ids) == [0, 1000])

    etype_ids, per_etype_ids = gpb.map_to_per_etype(
        F.tensor([0, 5000], dtype=edge_map_dtype)
    )
    assert etype_ids.dtype == edge_map_dtype
    assert per_etype_ids.dtype == edge_map_dtype
    assert np.all(F.asnumpy(etype_ids) == 0)
    assert np.all(F.asnumpy(per_etype_ids) == [0, 5000])

    node_policy = NodePartitionPolicy(gpb, DEFAULT_NTYPE)
    assert node_policy.type_name == DEFAULT_NTYPE
    edge_policy = EdgePartitionPolicy(gpb, DEFAULT_ETYPE)
    assert edge_policy.type_name == DEFAULT_ETYPE

    # Init via etype is not supported
    node_map = {
        "node1": F.tensor([[0, 1000], [1000, 2000]], dtype=node_map_dtype),
        "node2": F.tensor([[0, 1000], [1000, 2000]], dtype=node_map_dtype),
    }
    edge_map = {
        "edge1": F.tensor([[0, 5000], [5000, 10000]], dtype=edge_map_dtype)
    }
    ntypes = {"node1": 0, "node2": 1}
    etypes = {"edge1": 0}
    expect_except = False
    try:
        RangePartitionBook(
            part_id, num_parts, node_map, edge_map, ntypes, etypes
        )
    except AssertionError:
        expect_except = True
    assert expect_except
    expect_except = False
    try:
        EdgePartitionPolicy(gpb, "edge1")
    except AssertionError:
        expect_except = True
    assert expect_except

    # heterogeneous, init via canonical etype
    node_map = {
        "node1": F.tensor([[0, 1000], [1000, 2000]], dtype=node_map_dtype),
        "node2": F.tensor([[0, 1000], [1000, 2000]], dtype=node_map_dtype),
    }
    edge_map = {
        ("node1", "edge1", "node2"): F.tensor(
            [[0, 5000], [5000, 10000]], dtype=edge_map_dtype
        )
    }
    ntypes = {"node1": 0, "node2": 1}
    etypes = {("node1", "edge1", "node2"): 0}
    c_etype = list(etypes.keys())[0]
    gpb = RangePartitionBook(
        part_id, num_parts, node_map, edge_map, ntypes, etypes
    )
    assert gpb.etypes == ["edge1"]
    assert gpb.canonical_etypes == [c_etype]
    assert gpb.to_canonical_etype("edge1") == c_etype
    assert gpb.to_canonical_etype(c_etype) == c_etype

    ntype_ids, per_ntype_ids = gpb.map_to_per_ntype(
        F.tensor([0, 1000], dtype=node_map_dtype)
    )
    assert ntype_ids.dtype == node_map_dtype
    assert per_ntype_ids.dtype == node_map_dtype
    assert np.all(F.asnumpy(ntype_ids) == 0)
    assert np.all(F.asnumpy(per_ntype_ids) == [0, 1000])

    etype_ids, per_etype_ids = gpb.map_to_per_etype(
        F.tensor([0, 5000], dtype=edge_map_dtype)
    )
    assert etype_ids.dtype == edge_map_dtype
    assert per_etype_ids.dtype == edge_map_dtype
    assert np.all(F.asnumpy(etype_ids) == 0)
    assert np.all(F.asnumpy(per_etype_ids) == [0, 5000])

    expect_except = False
    try:
        gpb.to_canonical_etype(("node1", "edge2", "node2"))
    except BaseException:
        expect_except = True
    assert expect_except
    expect_except = False
    try:
        gpb.to_canonical_etype("edge2")
    except BaseException:
        expect_except = True
    assert expect_except

    # NodePartitionPolicy
    node_policy = NodePartitionPolicy(gpb, "node1")
    assert node_policy.type_name == "node1"
    assert node_policy.policy_str == "node~node1"
    assert node_policy.part_id == part_id
    assert node_policy.is_node
    assert node_policy.get_data_name("x").is_node()
    local_ids = th.arange(0, 1000)
    global_ids = local_ids + 1000
    assert th.equal(node_policy.to_local(global_ids), local_ids)
    assert th.all(node_policy.to_partid(global_ids) == part_id)
    assert node_policy.get_part_size() == 1000
    assert node_policy.get_size() == 2000

    # EdgePartitionPolicy
    edge_policy = EdgePartitionPolicy(gpb, c_etype)
    assert edge_policy.type_name == c_etype
    assert edge_policy.policy_str == "edge~node1:edge1:node2"
    assert edge_policy.part_id == part_id
    assert not edge_policy.is_node
    assert not edge_policy.get_data_name("x").is_node()
    local_ids = th.arange(0, 5000)
    global_ids = local_ids + 5000
    assert th.equal(edge_policy.to_local(global_ids), local_ids)
    assert th.all(edge_policy.to_partid(global_ids) == part_id)
    assert edge_policy.get_part_size() == 5000
    assert edge_policy.get_size() == 10000

    expect_except = False
    try:
        HeteroDataName(False, "edge1", "feat")
    except BaseException:
        expect_except = True
    assert expect_except
    data_name = HeteroDataName(False, c_etype, "feat")
    assert data_name.get_type() == c_etype


def test_UnknownPartitionBook():
    node_map = {"_N": {0: 0, 1: 1, 2: 2}}
    edge_map = {"_N:_E:_N": {0: 0, 1: 1, 2: 2}}

    part_metadata = {
        "num_parts": 1,
        "num_nodes": len(node_map),
        "num_edges": len(edge_map),
        "node_map": node_map,
        "edge_map": edge_map,
        "graph_name": "test_graph",
    }

    with tempfile.TemporaryDirectory() as test_dir:
        part_config = os.path.join(test_dir, "test_graph.json")
        with open(part_config, "w") as file:
            json.dump(part_metadata, file, indent=4)
        try:
            load_partition_book(part_config, 0)
        except Exception as e:
            if not isinstance(e, TypeError):
                raise e


@pytest.mark.parametrize("part_method", ["metis", "random"])
@pytest.mark.parametrize("num_parts", [1, 4])
@pytest.mark.parametrize("store_eids", [True, False])
@pytest.mark.parametrize("store_inner_node", [True, False])
@pytest.mark.parametrize("store_inner_edge", [True, False])
@pytest.mark.parametrize("debug_mode", [True, False])
def test_dgl_partition_to_graphbolt_homo(
    part_method,
    num_parts,
    store_eids,
    store_inner_node,
    store_inner_edge,
    debug_mode,
):
    reset_envs()
    if debug_mode:
        os.environ["DGL_DIST_DEBUG"] = "1"
    with tempfile.TemporaryDirectory() as test_dir:
        g = create_random_graph(1000)
        graph_name = "test"
        partition_graph(
            g, graph_name, num_parts, test_dir, part_method=part_method
        )
        part_config = os.path.join(test_dir, f"{graph_name}.json")
        dgl_partition_to_graphbolt(
            part_config,
            store_eids=store_eids,
            store_inner_node=store_inner_node,
            store_inner_edge=store_inner_edge,
        )
        for part_id in range(num_parts):
            orig_g = dgl.load_graphs(
                os.path.join(test_dir, f"part{part_id}/graph.dgl")
            )[0][0]
            os.remove(os.path.join(test_dir, f"part{part_id}/graph.dgl"))
            new_g = load_partition(
                part_config, part_id, load_feats=False, use_graphbolt=True
            )[0]
            orig_indptr, orig_indices, orig_eids = orig_g.adj().csc()
            # The original graph is in int64 while the partitioned graph is in
            # int32 as dtype formatting is applied when converting to graphbolt
            # format.
            assert orig_indptr.dtype == th.int64
            assert orig_indices.dtype == th.int64
            assert new_g.csc_indptr.dtype == th.int32
            assert new_g.indices.dtype == th.int32
            assert th.equal(orig_indptr, new_g.csc_indptr)
            assert th.equal(orig_indices, new_g.indices)
            assert new_g.node_type_offset is None
            assert orig_g.ndata[dgl.NID].dtype == th.int64
            assert new_g.node_attributes[dgl.NID].dtype == th.int64
            assert th.equal(
                orig_g.ndata[dgl.NID], new_g.node_attributes[dgl.NID]
            )
            if store_inner_node or debug_mode:
                assert th.equal(
                    orig_g.ndata["inner_node"],
                    new_g.node_attributes["inner_node"],
                )
            if store_eids or debug_mode:
                assert orig_g.edata[dgl.EID].dtype == th.int64
                assert new_g.edge_attributes[dgl.EID].dtype == th.int64
                assert th.equal(
                    orig_g.edata[dgl.EID][orig_eids],
                    new_g.edge_attributes[dgl.EID],
                )
            if store_inner_edge or debug_mode:
                assert orig_g.edata["inner_edge"].dtype == th.uint8
                assert new_g.edge_attributes["inner_edge"].dtype == th.uint8
                assert th.equal(
                    orig_g.edata["inner_edge"][orig_eids],
                    new_g.edge_attributes["inner_edge"],
                )
            assert new_g.type_per_edge is None
            assert new_g.node_type_to_id is None
            assert new_g.edge_type_to_id is None


@pytest.mark.parametrize("part_method", ["metis", "random"])
@pytest.mark.parametrize("num_parts", [1, 4])
@pytest.mark.parametrize("store_eids", [True, False])
@pytest.mark.parametrize("store_inner_node", [True, False])
@pytest.mark.parametrize("store_inner_edge", [True, False])
@pytest.mark.parametrize("debug_mode", [True, False])
def test_dgl_partition_to_graphbolt_hetero(
    part_method,
    num_parts,
    store_eids,
    store_inner_node,
    store_inner_edge,
    debug_mode,
):
    reset_envs()
    if debug_mode:
        os.environ["DGL_DIST_DEBUG"] = "1"
    with tempfile.TemporaryDirectory() as test_dir:
        g = create_random_hetero()
        graph_name = "test"
        partition_graph(
            g, graph_name, num_parts, test_dir, part_method=part_method
        )
        part_config = os.path.join(test_dir, f"{graph_name}.json")
        dgl_partition_to_graphbolt(
            part_config,
            store_eids=store_eids,
            store_inner_node=store_inner_node,
            store_inner_edge=store_inner_edge,
        )
        for part_id in range(num_parts):
            orig_g = dgl.load_graphs(
                os.path.join(test_dir, f"part{part_id}/graph.dgl")
            )[0][0]
            os.remove(os.path.join(test_dir, f"part{part_id}/graph.dgl"))
            new_g = load_partition(
                part_config, part_id, load_feats=False, use_graphbolt=True
            )[0]
            orig_indptr, orig_indices, orig_eids = orig_g.adj().csc()

            # Edges should be sorted in etype for the same dst node.
            if debug_mode:
                num_inner_edges = orig_g.edata["inner_edge"].sum().item()
                assert (
                    num_inner_edges
                    == orig_g.edata["inner_edge"][th.arange(num_inner_edges)]
                    .sum()
                    .item()
                )
                assert (
                    num_inner_edges
                    == new_g.edge_attributes["inner_edge"][:num_inner_edges]
                    .sum()
                    .item()
                )
                num_inner_nodes = orig_g.ndata["inner_node"].sum().item()
                assert (
                    num_inner_nodes
                    == orig_g.ndata["inner_node"][th.arange(num_inner_nodes)]
                    .sum()
                    .item()
                )
                assert (
                    num_inner_nodes
                    == new_g.node_attributes["inner_node"][:num_inner_nodes]
                    .sum()
                    .item()
                )
                for i in range(orig_g.num_nodes()):
                    if orig_g.in_degrees(i) == 0:
                        continue
                    # Verify DGLGraph partitions.
                    eids = orig_g.in_edges(i, form="eid")
                    etypes = orig_g.edata[dgl.ETYPE][eids]
                    assert th.equal(etypes, etypes.sort()[0])
                    # Verify GraphBolt partitions.
                    eids_start = new_g.csc_indptr[i]
                    eids_end = new_g.csc_indptr[i + 1]
                    etypes = new_g.edge_attributes[dgl.ETYPE][
                        eids_start:eids_end
                    ]
                    assert th.equal(etypes, etypes.sort()[0])

            # The original graph is in int64 while the partitioned graph is in
            # int32 as dtype formatting is applied when converting to graphbolt
            # format.
            assert orig_indptr.dtype == th.int64
            assert orig_indices.dtype == th.int64
            assert new_g.csc_indptr.dtype == th.int32
            assert new_g.indices.dtype == th.int32
            assert th.equal(orig_indptr, new_g.csc_indptr)
            assert th.equal(orig_indices, new_g.indices)
            assert orig_g.ndata[dgl.NID].dtype == th.int64
            assert new_g.node_attributes[dgl.NID].dtype == th.int64
            assert th.equal(
                orig_g.ndata[dgl.NID], new_g.node_attributes[dgl.NID]
            )
            if store_inner_node or debug_mode:
                assert th.equal(
                    orig_g.ndata["inner_node"],
                    new_g.node_attributes["inner_node"],
                )
            if debug_mode:
                assert orig_g.ndata[dgl.NTYPE].dtype == th.int32
                assert new_g.node_attributes[dgl.NTYPE].dtype == th.int8
                assert th.equal(
                    orig_g.ndata[dgl.NTYPE], new_g.node_attributes[dgl.NTYPE]
                )
            if store_eids or debug_mode:
                assert orig_g.edata[dgl.EID].dtype == th.int64
                assert new_g.edge_attributes[dgl.EID].dtype == th.int64
                assert th.equal(
                    orig_g.edata[dgl.EID][orig_eids],
                    new_g.edge_attributes[dgl.EID],
                )
            if store_inner_edge or debug_mode:
                assert orig_g.edata["inner_edge"].dtype == th.uint8
                assert new_g.edge_attributes["inner_edge"].dtype == th.uint8
                assert th.equal(
                    orig_g.edata["inner_edge"],
                    new_g.edge_attributes["inner_edge"],
                )
            if debug_mode:
                assert orig_g.edata[dgl.ETYPE].dtype == th.int32
                assert new_g.edge_attributes[dgl.ETYPE].dtype == th.int8
                assert th.equal(
                    orig_g.edata[dgl.ETYPE][orig_eids],
                    new_g.edge_attributes[dgl.ETYPE],
                )
            assert th.equal(
                orig_g.edata[dgl.ETYPE][orig_eids], new_g.type_per_edge
            )

            for node_type, type_id in new_g.node_type_to_id.items():
                assert g.get_ntype_id(node_type) == type_id
            for edge_type, type_id in new_g.edge_type_to_id.items():
                assert g.get_etype_id(_etype_str_to_tuple(edge_type)) == type_id
            assert new_g.node_type_offset is None


def test_not_sorted_node_edge_map():
    # Partition configure file which includes not sorted node/edge map.
    part_config_str = """
{
    "edge_map": {
        "item:likes-rev:user": [
            [
                0,
                100
            ],
            [
                1000,
                1500
            ]
        ],
        "user:follows-rev:user": [
            [
                300,
                600
            ],
            [
                2100,
                2800
            ]
        ],
        "user:follows:user": [
            [
                100,
                300
            ],
            [
                1500,
                2100
            ]
        ],
        "user:likes:item": [
            [
                600,
                1000
            ],
            [
                2800,
                3600
            ]
        ]
    },
    "etypes": {
        "item:likes-rev:user": 0,
        "user:follows-rev:user": 2,
        "user:follows:user": 1,
        "user:likes:item": 3
    },
    "graph_name": "test_graph",
    "halo_hops": 1,
    "node_map": {
        "user": [
            [
                100,
                300
            ],
            [
                600,
                1000
            ]
        ],
        "item": [
            [
                0,
                100
            ],
            [
                300,
                600
            ]
        ]
    },
    "ntypes": {
        "user": 1,
        "item": 0
    },
    "num_edges": 3600,
    "num_nodes": 1000,
    "num_parts": 2,
    "part-0": {
        "edge_feats": "part0/edge_feat.dgl",
        "node_feats": "part0/node_feat.dgl",
        "part_graph": "part0/graph.dgl"
    },
    "part-1": {
        "edge_feats": "part1/edge_feat.dgl",
        "node_feats": "part1/node_feat.dgl",
        "part_graph": "part1/graph.dgl"
    },
    "part_method": "metis"
}
    """
    with tempfile.TemporaryDirectory() as test_dir:
        part_config = os.path.join(test_dir, "test_graph.json")
        with open(part_config, "w") as file:
            file.write(part_config_str)
        # Part 0.
        gpb, _, _, _ = load_partition_book(part_config, 0)
        assert gpb.local_ntype_offset == [0, 100, 300]
        assert gpb.local_etype_offset == [0, 100, 300, 600, 1000]
        # Patr 1.
        gpb, _, _, _ = load_partition_book(part_config, 1)
        assert gpb.local_ntype_offset == [0, 300, 700]
        assert gpb.local_etype_offset == [0, 500, 1100, 1800, 2600]


def _get_part_IDs(part_g):
    # These are partition-local IDs.
    num_columns = part_g.csc_indptr.diff()
    part_src_ids = part_g.indices
    part_dst_ids = th.arange(part_g.total_num_nodes).repeat_interleave(
        num_columns
    )
    # These are reshuffled global homogeneous IDs.
    part_src_ids = F.gather_row(part_g.node_attributes[dgl.NID], part_src_ids)
    part_dst_ids = F.gather_row(part_g.node_attributes[dgl.NID], part_dst_ids)
    return part_src_ids, part_dst_ids


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


def _verify_metadata_gb(gpb, g, num_parts, part_id, part_sizes):
    """
    check list:
        make sure the number of nodes and edges is correct.
        make sure the number of parts is correct.
        make sure the number of nodes and edges in each parts os corrcet.
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


@pytest.mark.parametrize("part_method", ["metis", "random"])
@pytest.mark.parametrize("num_parts", [1, 4])
@pytest.mark.parametrize("store_eids", [True, False])
@pytest.mark.parametrize("store_inner_node", [True, False])
@pytest.mark.parametrize("store_inner_edge", [True, False])
@pytest.mark.parametrize("debug_mode", [True, False])
def test_partition_graph_graphbolt_homo(
    part_method,
    num_parts,
    store_eids,
    store_inner_node,
    store_inner_edge,
    debug_mode,
):
    reset_envs()
    if debug_mode:
        os.environ["DGL_DIST_DEBUG"] = "1"
    with tempfile.TemporaryDirectory() as test_dir:
        g = create_random_graph(1000)
        graph_name = "test"
        g.ndata["labels"] = F.arange(0, g.num_nodes())
        g.ndata["feats"] = F.tensor(
            np.random.randn(g.num_nodes(), 10), F.float32
        )
        g.edata["feats"] = F.tensor(
            np.random.randn(g.num_edges(), 10), F.float32
        )

        orig_nids, orig_eids = partition_graph(
            g,
            graph_name,
            num_parts,
            test_dir,
            part_method=part_method,
            use_graphbolt=True,
            store_eids=store_eids,
            store_inner_node=store_inner_node,
            store_inner_edge=store_inner_edge,
            return_mapping=True,
        )

        if debug_mode:
            store_eids = store_inner_node = store_inner_edge = True

        _verify_graphbolt_part(
            g,
            test_dir,
            orig_nids,
            orig_eids,
            graph_name,
            num_parts,
            store_inner_node,
            store_inner_edge,
            store_eids,
            is_homo=True,
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
        else g.nodes[test_ntype].data["labels"]
    )
    edata = (
        g.edata["feats"]
        if test_etype is None
        else g.edges[test_etype].data["labels"]
    )

    orig_labels[F.asnumpy(orig_nid)] = shuffled_labels
    orig_edata[F.asnumpy(orig_eid)] = shuffled_edata
    assert np.all(orig_labels == F.asnumpy(nlabel))
    assert np.all(orig_edata == F.asnumpy(edata))


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
        shuffled_labels.append(node_feats[test_ntype + "/labels"])
        shuffled_edata.append(
            edge_feats[_etype_tuple_to_str(test_etype) + "/labels"]
        )
    else:
        shuffled_labels.append(node_feats["_N/labels"])
        shuffled_edata.append(edge_feats["_N:_E:_N/feats"])
    parts.append(part_g)
    gpbs.append(gpb)
    tot_node_feats.append(node_feats)
    tot_edge_feats.append(edge_feats)


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


def _verify_original_IDs_type_hetero(hg, orig_nids, orig_eids):
    """
    check list:
        make sure type of nodes and edges' ids are correct.
        make sure nodes and edges' number in each type is correct.
    """
    assert len(orig_nids) == len(hg.ntypes)
    assert len(orig_eids) == len(hg.canonical_etypes)
    for ntype in hg.ntypes:
        assert len(orig_nids[ntype]) == hg.num_nodes(ntype)
        assert F.dtype(orig_nids[ntype]) in (F.int64, F.int32)
    for etype in hg.canonical_etypes:
        assert len(orig_eids[etype]) == hg.num_edges(etype)
        assert F.dtype(orig_eids[etype]) in (F.int64, F.int32)


@pytest.mark.parametrize("part_method", ["metis", "random"])
@pytest.mark.parametrize("num_parts", [1, 4])
@pytest.mark.parametrize("store_eids", [True, False])
@pytest.mark.parametrize("store_inner_node", [True, False])
@pytest.mark.parametrize("store_inner_edge", [True, False])
@pytest.mark.parametrize("debug_mode", [True, False])
def test_partition_graph_graphbolt_hetero(
    part_method,
    num_parts,
    store_eids,
    store_inner_node,
    store_inner_edge,
    debug_mode,
    n_jobs=1,
):
    test_ntype = "n1"
    test_etype = ("n1", "r1", "n2")
    reset_envs()
    if debug_mode:
        os.environ["DGL_DIST_DEBUG"] = "1"
    with tempfile.TemporaryDirectory() as test_dir:
        hg = create_random_hetero()
        graph_name = "test"
        hg.nodes[test_ntype].data["labels"] = F.arange(
            0, hg.num_nodes(test_ntype)
        )
        hg.nodes[test_ntype].data["feats"] = F.tensor(
            np.random.randn(hg.num_nodes(test_ntype), 10), F.float32
        )
        hg.edges[test_etype].data["feats"] = F.tensor(
            np.random.randn(hg.num_edges(test_etype), 10), F.float32
        )
        hg.edges[test_etype].data["labels"] = F.arange(
            0, hg.num_edges(test_etype)
        )
        orig_nids, orig_eids = partition_graph(
            hg,
            graph_name,
            num_parts,
            test_dir,
            part_method=part_method,
            return_mapping=True,
            num_trainers_per_machine=1,
            use_graphbolt=True,
            store_eids=store_eids,
            store_inner_node=store_inner_node,
            store_inner_edge=store_inner_edge,
            n_jobs=n_jobs,
        )

        _verify_original_IDs_type_hetero(hg, orig_nids, orig_eids)
        if debug_mode:
            store_eids = store_inner_node = store_inner_edge = True

        parts = _verify_graphbolt_part(
            hg,
            test_dir,
            orig_nids,
            orig_eids,
            graph_name,
            num_parts,
            store_inner_node,
            store_inner_edge,
            store_eids,
            test_ntype,
            test_etype,
            is_homo=False,
        )

        _verify_hetero_graph(
            hg,
            parts,
            store_eids=store_eids,
            store_inner_edge=store_inner_edge,
            debug_mode=debug_mode,
        )


@pytest.mark.parametrize("part_method", ["metis", "random"])
@pytest.mark.parametrize("num_parts", [1, 4])
@pytest.mark.parametrize("graph_formats", [["csc"], ["coo"], ["coo", "csc"]])
def test_partition_graph_graphbolt_homo_find_edges(
    part_method,
    num_parts,
    graph_formats,
    n_jobs=1,
):
    reset_envs()
    os.environ["DGL_DIST_DEBUG"] = "1"
    with tempfile.TemporaryDirectory() as test_dir:
        g = create_random_graph(1000)
        g.ndata["feat"] = th.rand(g.num_nodes(), 5)
        graph_name = "test"
        orig_nids, orig_eids = partition_graph(
            g,
            graph_name,
            num_parts,
            test_dir,
            part_method=part_method,
            graph_formats=graph_formats,
            return_mapping=True,
            use_graphbolt=True,
            store_eids=True,
            store_inner_node=True,
            store_inner_edge=True,
            n_jobs=n_jobs,
        )
        part_config = os.path.join(test_dir, f"{graph_name}.json")
        for part_id in range(num_parts):
            local_g, _, _, gpb, _, _, _ = load_partition(
                part_config, part_id, load_feats=False, use_graphbolt=True
            )
            inner_local_eids = th.nonzero(
                local_g.edge_attributes["inner_edge"], as_tuple=False
            ).squeeze()
            inner_global_eids = local_g.edge_attributes[dgl.EID][
                inner_local_eids
            ]
            if "coo" not in graph_formats:
                with pytest.raises(
                    ValueError,
                    match="The edge attributes DGL2GB_EID and GB_DST_ID are "
                    "not found. Please make sure `coo` format is available"
                    " when generating partitions in GraphBolt format.",
                ):
                    dgl.distributed.graph_services._find_edges(
                        local_g, gpb, inner_global_eids
                    )
                continue
            global_src, global_dst = dgl.distributed.graph_services._find_edges(
                local_g, gpb, inner_global_eids
            )
            orig_global_src = orig_nids[global_src]
            orig_global_dst = orig_nids[global_dst]
            assert th.all(g.has_edges_between(orig_global_src, orig_global_dst))

            # dtype check.
            assert (
                local_g.edge_attributes[dgl.distributed.DGL2GB_EID].dtype
                == th.int32
            )
            assert (
                local_g.edge_attributes[dgl.distributed.GB_DST_ID].dtype
                == th.int32
            )

            # No need to map local node IDs.
            inner_local_nids = th.nonzero(
                local_g.node_attributes["inner_node"], as_tuple=False
            ).squeeze()
            inner_global_nids = local_g.node_attributes[dgl.NID][
                inner_local_nids
            ]
            assert th.equal(
                inner_local_nids, gpb.nid2localnid(inner_global_nids, part_id)
            )

            # Need to map local edge IDs.
            DGL_inner_local_eids = gpb.eid2localeid(inner_global_eids, part_id)
            GB_inner_local_eids = local_g.edge_attributes[
                dgl.distributed.DGL2GB_EID
            ][DGL_inner_local_eids]
            assert th.equal(inner_local_eids, GB_inner_local_eids)


@pytest.mark.parametrize("part_method", ["metis", "random"])
@pytest.mark.parametrize("num_parts", [1, 4])
@pytest.mark.parametrize("graph_formats", [["csc"], ["coo"], ["coo", "csc"]])
def test_partition_graph_graphbolt_hetero_find_edges(
    part_method,
    num_parts,
    graph_formats,
    n_jobs=1,
):
    reset_envs()
    os.environ["DGL_DIST_DEBUG"] = "1"
    with tempfile.TemporaryDirectory() as test_dir:
        hg = create_random_hetero()
        graph_name = "test"
        orig_nids, orig_eids = partition_graph(
            hg,
            graph_name,
            num_parts,
            test_dir,
            part_method=part_method,
            graph_formats=graph_formats,
            return_mapping=True,
            use_graphbolt=True,
            store_eids=True,
            store_inner_node=True,
            store_inner_edge=True,
            n_jobs=n_jobs,
        )
        part_config = os.path.join(test_dir, f"{graph_name}.json")
        for part_id in range(num_parts):
            local_g, _, _, gpb, _, _, _ = load_partition(
                part_config, part_id, load_feats=False, use_graphbolt=True
            )
            inner_local_eids = th.nonzero(
                local_g.edge_attributes["inner_edge"], as_tuple=False
            ).squeeze()
            inner_global_eids = local_g.edge_attributes[dgl.EID][
                inner_local_eids
            ]
            if "coo" not in graph_formats:
                with pytest.raises(
                    ValueError,
                    match="The edge attributes DGL2GB_EID and GB_DST_ID are "
                    "not found. Please make sure `coo` format is available"
                    " when generating partitions in GraphBolt format.",
                ):
                    dgl.distributed.graph_services._find_edges(
                        local_g, gpb, inner_global_eids
                    )
                continue
            global_src, global_dst = dgl.distributed.graph_services._find_edges(
                local_g, gpb, inner_global_eids
            )
            ntype_ids_src, per_ntype_nids_src = gpb.map_to_per_ntype(global_src)
            ntype_ids_dst, per_ntype_nids_dst = gpb.map_to_per_ntype(global_dst)
            etype_ids, per_etype_eids = gpb.map_to_per_etype(inner_global_eids)
            for src_ntype, etype, dst_ntype in hg.canonical_etypes:
                etype_id = hg.get_etype_id((src_ntype, etype, dst_ntype))
                current_etype_indices = th.nonzero(
                    etype_ids == etype_id, as_tuple=False
                ).squeeze()
                assert th.all(
                    ntype_ids_src[current_etype_indices]
                    == gpb.ntypes.index(src_ntype)
                )
                assert th.all(
                    ntype_ids_dst[current_etype_indices]
                    == gpb.ntypes.index(dst_ntype)
                )
                current_per_ntype_nids_src = per_ntype_nids_src[
                    current_etype_indices
                ]
                current_per_ntype_nids_dst = per_ntype_nids_dst[
                    current_etype_indices
                ]
                current_orig_global_src = orig_nids[src_ntype][
                    current_per_ntype_nids_src
                ]
                current_orig_global_dst = orig_nids[dst_ntype][
                    current_per_ntype_nids_dst
                ]
                assert th.all(
                    hg.has_edges_between(
                        current_orig_global_src,
                        current_orig_global_dst,
                        etype=(src_ntype, etype, dst_ntype),
                    )
                )
                current_orig_global_eids = orig_eids[
                    (src_ntype, etype, dst_ntype)
                ][per_etype_eids[current_etype_indices]]
                orig_src_ids, orig_dst_ids = hg.find_edges(
                    current_orig_global_eids,
                    etype=(src_ntype, etype, dst_ntype),
                )
                assert th.equal(current_orig_global_src, orig_src_ids)
                assert th.equal(current_orig_global_dst, orig_dst_ids)

            # dtype check.
            assert (
                local_g.edge_attributes[dgl.distributed.DGL2GB_EID].dtype
                == th.int32
            )
            assert (
                local_g.edge_attributes[dgl.distributed.GB_DST_ID].dtype
                == th.int32
            )

            # No need to map local node IDs.
            inner_local_nids = th.nonzero(
                local_g.node_attributes["inner_node"], as_tuple=False
            ).squeeze()
            inner_global_nids = local_g.node_attributes[dgl.NID][
                inner_local_nids
            ]
            assert th.equal(
                inner_local_nids, gpb.nid2localnid(inner_global_nids, part_id)
            )

            # Need to map local edge IDs.
            DGL_inner_local_eids = gpb.eid2localeid(inner_global_eids, part_id)
            GB_inner_local_eids = local_g.edge_attributes[
                dgl.distributed.DGL2GB_EID
            ][DGL_inner_local_eids]
            assert th.equal(inner_local_eids, GB_inner_local_eids)


@pytest.mark.parametrize("num_parts", [1, 4])
def test_partition_graph_graphbolt_hetero_multi(
    num_parts,
):
    reset_envs()

    test_partition_graph_graphbolt_hetero(
        part_method="random",
        num_parts=num_parts,
        n_jobs=4,
        store_eids=True,
        store_inner_node=True,
        store_inner_edge=True,
        debug_mode=False,
    )


@pytest.mark.parametrize("num_parts", [1, 4])
def test_partition_graph_graphbolt_homo_find_edges_multi(
    num_parts,
):
    test_partition_graph_graphbolt_homo_find_edges(
        part_method="random",
        num_parts=num_parts,
        graph_formats="coo",
        n_jobs=4,
    )


@pytest.mark.parametrize("num_parts", [1, 4])
def test_partition_graph_graphbolt_hetero_find_edges_multi(
    num_parts,
):
    test_partition_graph_graphbolt_hetero_find_edges(
        part_method="random",
        num_parts=num_parts,
        graph_formats="coo",
        n_jobs=4,
    )


@pytest.mark.parametrize("part_method", ["metis", "random"])
@pytest.mark.parametrize("num_parts", [4])
@pytest.mark.parametrize("num_trainers_per_machine", [1])
@pytest.mark.parametrize("graph_formats", [None])
def test_partition_hetero_few_edges(
    part_method,
    num_parts,
    num_trainers_per_machine,
    graph_formats,
):
    os.environ["DGL_DIST_DEBUG"] = "1"
    if part_method == "random" and num_parts > 1:
        num_trainers_per_machine = 1

    # Create a heterograph with 2 edges for one edge type.
    hg = create_random_hetero()
    edges_coo = {
        c_etype: hg.edges(etype=c_etype) for c_etype in hg.canonical_etypes
    }
    edges_coo[("n1", "a0", "n2")] = (th.tensor([0, 1]), th.tensor([1, 0]))
    edges_coo[("n1", "a1", "n3")] = (th.tensor([0, 1]), th.tensor([1, 0]))
    hg = dgl.heterograph(edges_coo)

    check_hetero_partition(
        hg,
        part_method,
        num_parts,
        num_trainers_per_machine,
        load_feats=False,
        graph_formats=graph_formats,
    )
    reset_envs()


@pytest.mark.parametrize("part_method", ["metis", "random"])
@pytest.mark.parametrize("num_parts", [4])
@pytest.mark.parametrize("num_trainers_per_machine", [1])
@pytest.mark.parametrize("graph_formats", [None])
def test_partition_hetero_few_nodes(
    part_method,
    num_parts,
    num_trainers_per_machine,
    graph_formats,
):
    os.environ["DGL_DIST_DEBUG"] = "1"
    if part_method == "random" and num_parts > 1:
        num_trainers_per_machine = 1

    # Create a heterograph with 2 nodes for one node type.
    hg = create_random_hetero()
    edges_coo = {
        c_etype: hg.edges(etype=c_etype) for c_etype in hg.canonical_etypes
    }
    edges_coo[("n1", "r_few", "n_few")] = (th.tensor([0, 1]), th.tensor([1, 0]))
    edges_coo[("a0", "a01", "n_1")] = (th.tensor([0, 1]), th.tensor([1, 0]))
    hg = dgl.heterograph(edges_coo)

    expected_exception = False
    try:
        check_hetero_partition(
            hg,
            part_method,
            num_parts,
            num_trainers_per_machine,
            load_feats=False,
            graph_formats=graph_formats,
        )
    except Exception as e:
        expected_exception = True
    assert expected_exception == (part_method == "metis")
    reset_envs()
