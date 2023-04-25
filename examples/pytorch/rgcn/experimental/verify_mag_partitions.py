import json
import os

import dgl
import numpy as np
import torch as th
from ogb.nodeproppred import DglNodePropPredDataset

partitions_folder = "outputs"
graph_name = "mag"
with open("{}/{}.json".format(partitions_folder, graph_name)) as json_file:
    metadata = json.load(json_file)
num_parts = metadata["num_parts"]

# Load OGB-MAG.
dataset = DglNodePropPredDataset(name="ogbn-mag")
hg_orig, labels = dataset[0]
subgs = {}
for etype in hg_orig.canonical_etypes:
    u, v = hg_orig.all_edges(etype=etype)
    subgs[etype] = (u, v)
    subgs[(etype[2], "rev-" + etype[1], etype[0])] = (v, u)
hg = dgl.heterograph(subgs)
hg.nodes["paper"].data["feat"] = hg_orig.nodes["paper"].data["feat"]

# Construct node data and edge data after reshuffling.
node_feats = {}
edge_feats = {}
for partid in range(num_parts):
    part_node_feats = dgl.data.utils.load_tensors(
        "{}/part{}/node_feat.dgl".format(partitions_folder, partid)
    )
    part_edge_feats = dgl.data.utils.load_tensors(
        "{}/part{}/edge_feat.dgl".format(partitions_folder, partid)
    )
    for key in part_node_feats:
        if key in node_feats:
            node_feats[key].append(part_node_feats[key])
        else:
            node_feats[key] = [part_node_feats[key]]
    for key in part_edge_feats:
        if key in edge_feats:
            edge_feats[key].append(part_edge_feats[key])
        else:
            edge_feats[key] = [part_edge_feats[key]]
for key in node_feats:
    node_feats[key] = th.cat(node_feats[key])
for key in edge_feats:
    edge_feats[key] = th.cat(edge_feats[key])

ntype_map = metadata["ntypes"]
ntypes = [None] * len(ntype_map)
for key in ntype_map:
    ntype_id = ntype_map[key]
    ntypes[ntype_id] = key
etype_map = metadata["etypes"]
etypes = [None] * len(etype_map)
for key in etype_map:
    etype_id = etype_map[key]
    etypes[etype_id] = key

etype2canonical = {
    etype: (srctype, etype, dsttype)
    for srctype, etype, dsttype in hg.canonical_etypes
}

node_map = metadata["node_map"]
for key in node_map:
    node_map[key] = th.stack([th.tensor(row) for row in node_map[key]], 0)
nid_map = dgl.distributed.id_map.IdMap(node_map)
edge_map = metadata["edge_map"]
for key in edge_map:
    edge_map[key] = th.stack([th.tensor(row) for row in edge_map[key]], 0)
eid_map = dgl.distributed.id_map.IdMap(edge_map)

for ntype in node_map:
    assert hg.num_nodes(ntype) == th.sum(
        node_map[ntype][:, 1] - node_map[ntype][:, 0]
    )
for etype in edge_map:
    assert hg.num_edges(etype) == th.sum(
        edge_map[etype][:, 1] - edge_map[etype][:, 0]
    )

# verify part_0 with graph_partition_book
eid = []
gpb = dgl.distributed.graph_partition_book.RangePartitionBook(
    0,
    num_parts,
    node_map,
    edge_map,
    {ntype: i for i, ntype in enumerate(hg.ntypes)},
    {etype: i for i, etype in enumerate(hg.etypes)},
)
subg0 = dgl.load_graphs("{}/part0/graph.dgl".format(partitions_folder))[0][0]
for etype in hg.etypes:
    type_eid = th.zeros((1,), dtype=th.int64)
    eid.append(gpb.map_to_homo_eid(type_eid, etype))
eid = th.cat(eid)
part_id = gpb.eid2partid(eid)
assert th.all(part_id == 0)
local_eid = gpb.eid2localeid(eid, 0)
assert th.all(local_eid == eid)
assert th.all(subg0.edata[dgl.EID][local_eid] == eid)
lsrc, ldst = subg0.find_edges(local_eid)
gsrc, gdst = subg0.ndata[dgl.NID][lsrc], subg0.ndata[dgl.NID][ldst]
# The destination nodes are owned by the partition.
assert th.all(gdst == ldst)
# gdst which is not assigned into current partition is not required to equal ldst
assert th.all(th.logical_or(gdst == ldst, subg0.ndata["inner_node"][ldst] == 0))
etids, _ = gpb.map_to_per_etype(eid)
src_tids, _ = gpb.map_to_per_ntype(gsrc)
dst_tids, _ = gpb.map_to_per_ntype(gdst)
canonical_etypes = []
etype_ids = th.arange(0, len(etypes))
for src_tid, etype_id, dst_tid in zip(src_tids, etype_ids, dst_tids):
    canonical_etypes.append(
        (ntypes[src_tid], etypes[etype_id], ntypes[dst_tid])
    )
for etype in canonical_etypes:
    assert etype in hg.canonical_etypes

# Load the graph partition structure.
orig_node_ids = {ntype: [] for ntype in hg.ntypes}
orig_edge_ids = {etype: [] for etype in hg.etypes}
for partid in range(num_parts):
    print("test part", partid)
    part_file = "{}/part{}/graph.dgl".format(partitions_folder, partid)
    subg = dgl.load_graphs(part_file)[0][0]
    subg_src_id, subg_dst_id = subg.edges()
    orig_src_id = subg.ndata["orig_id"][subg_src_id]
    orig_dst_id = subg.ndata["orig_id"][subg_dst_id]
    global_src_id = subg.ndata[dgl.NID][subg_src_id]
    global_dst_id = subg.ndata[dgl.NID][subg_dst_id]
    subg_ntype = subg.ndata[dgl.NTYPE]
    subg_etype = subg.edata[dgl.ETYPE]
    for ntype_id in th.unique(subg_ntype):
        ntype = ntypes[ntype_id]
        idx = subg_ntype == ntype_id
        # This is global IDs after reshuffle.
        nid = subg.ndata[dgl.NID][idx]
        ntype_ids1, type_nid = nid_map(nid)
        orig_type_nid = subg.ndata["orig_id"][idx]
        inner_node = subg.ndata["inner_node"][idx]
        # All nodes should have the same node type.
        assert np.all(ntype_ids1.numpy() == int(ntype_id))
        assert np.all(
            nid[inner_node == 1].numpy()
            == np.arange(node_map[ntype][partid, 0], node_map[ntype][partid, 1])
        )
        orig_node_ids[ntype].append(orig_type_nid[inner_node == 1])

        # Check the degree of the inner nodes.
        inner_nids = th.nonzero(
            th.logical_and(subg_ntype == ntype_id, subg.ndata["inner_node"]),
            as_tuple=True,
        )[0]
        subg_deg = subg.in_degrees(inner_nids)
        orig_nids = subg.ndata["orig_id"][inner_nids]
        # Calculate the in-degrees of nodes of a particular node type.
        glob_deg = th.zeros(len(subg_deg), dtype=th.int64)
        for etype in hg.canonical_etypes:
            dst_ntype = etype[2]
            if dst_ntype == ntype:
                glob_deg += hg.in_degrees(orig_nids, etype=etype)
        assert np.all(glob_deg.numpy() == subg_deg.numpy())

        # Check node data.
        for name in hg.nodes[ntype].data:
            local_data = node_feats[ntype + "/" + name][type_nid]
            local_data1 = hg.nodes[ntype].data[name][orig_type_nid]
            assert np.all(local_data.numpy() == local_data1.numpy())

    for etype_id in th.unique(subg_etype):
        etype = etypes[etype_id]
        srctype, _, dsttype = etype2canonical[etype]
        idx = subg_etype == etype_id
        exist = hg[etype].has_edges_between(orig_src_id[idx], orig_dst_id[idx])
        assert np.all(exist.numpy())
        eid = hg[etype].edge_ids(orig_src_id[idx], orig_dst_id[idx])
        assert np.all(eid.numpy() == subg.edata["orig_id"][idx].numpy())

        ntype_ids, type_nid = nid_map(global_src_id[idx])
        assert len(th.unique(ntype_ids)) == 1
        assert ntypes[ntype_ids[0]] == srctype
        ntype_ids, type_nid = nid_map(global_dst_id[idx])
        assert len(th.unique(ntype_ids)) == 1
        assert ntypes[ntype_ids[0]] == dsttype

        # This is global IDs after reshuffle.
        eid = subg.edata[dgl.EID][idx]
        etype_ids1, type_eid = eid_map(eid)
        orig_type_eid = subg.edata["orig_id"][idx]
        inner_edge = subg.edata["inner_edge"][idx]
        # All edges should have the same edge type.
        assert np.all(etype_ids1.numpy() == int(etype_id))
        assert np.all(
            np.sort(eid[inner_edge == 1].numpy())
            == np.arange(edge_map[etype][partid, 0], edge_map[etype][partid, 1])
        )
        orig_edge_ids[etype].append(orig_type_eid[inner_edge == 1])

        # Check edge data.
        for name in hg.edges[etype].data:
            local_data = edge_feats[etype + "/" + name][type_eid]
            local_data1 = hg.edges[etype].data[name][orig_type_eid]
            assert np.all(local_data.numpy() == local_data1.numpy())

for ntype in orig_node_ids:
    nids = th.cat(orig_node_ids[ntype])
    nids = th.sort(nids)[0]
    assert np.all((nids == th.arange(hg.num_nodes(ntype))).numpy())

for etype in orig_edge_ids:
    eids = th.cat(orig_edge_ids[etype])
    eids = th.sort(eids)[0]
    assert np.all((eids == th.arange(hg.num_edges(etype))).numpy())
