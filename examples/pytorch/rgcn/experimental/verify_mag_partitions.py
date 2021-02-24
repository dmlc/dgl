import os
import json
import numpy as np
import dgl
import torch as th
from ogb.nodeproppred import DglNodePropPredDataset

with open('outputs/mag.json') as json_file:
    metadata = json.load(json_file)
num_parts = metadata['num_parts']

# Load OGB-MAG.
dataset = DglNodePropPredDataset(name='ogbn-mag')
hg_orig, labels = dataset[0]
subgs = {}
for etype in hg_orig.canonical_etypes:
    u, v = hg_orig.all_edges(etype=etype)
    subgs[etype] = (u, v)
    subgs[(etype[2], 'rev-'+etype[1], etype[0])] = (v, u)
hg = dgl.heterograph(subgs)
hg.nodes['paper'].data['feat'] = hg_orig.nodes['paper'].data['feat']

# Construct node data and edge data after reshuffling.
node_feats = {}
edge_feats = {}
for partid in range(num_parts):
    part_node_feats = dgl.data.utils.load_tensors('outputs/part{}/node_feat.dgl'.format(partid))
    part_edge_feats = dgl.data.utils.load_tensors('outputs/part{}/edge_feat.dgl'.format(partid))
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

ntype_map = metadata['ntypes']
ntypes = [None] * len(ntype_map)
for key in ntype_map:
    ntype_id = ntype_map[key]
    ntypes[ntype_id] = key
etype_map = metadata['etypes']
etypes = [None] * len(etype_map)
for key in etype_map:
    etype_id = etype_map[key]
    etypes[etype_id] = key

node_map = metadata['node_map']
for key in node_map:
    node_map[key] = th.stack([th.tensor(row) for row in node_map[key]], 0)
nid_map = dgl.distributed.id_map.IdMap(node_map)
edge_map = metadata['edge_map']
for key in edge_map:
    edge_map[key] = th.stack([th.tensor(row) for row in edge_map[key]], 0)
eid_map = dgl.distributed.id_map.IdMap(edge_map)

# Load the graph partition structure.
for partid in range(num_parts):
    print('test part', partid)
    part_file = 'outputs/part{}/graph.dgl'.format(partid)
    subg = dgl.load_graphs(part_file)[0][0]
    subg_src_id, subg_dst_id = subg.edges()
    subg_src_id = subg.ndata['orig_id'][subg_src_id]
    subg_dst_id = subg.ndata['orig_id'][subg_dst_id]
    subg_ntype = subg.ndata[dgl.NTYPE]
    subg_etype = subg.edata[dgl.ETYPE]
    for ntype_id in th.unique(subg_ntype):
        ntype = ntypes[ntype_id]
        idx = subg_ntype == ntype_id
        # This is global IDs after reshuffle.
        nid = subg.ndata[dgl.NID][idx]
        ntype_ids1, type_nid = nid_map(nid)
        orig_type_nid = subg.ndata['orig_id'][idx]
        # All nodes should have the same node type.
        assert np.all(ntype_ids1.numpy() == int(ntype_id))

        # Check node data.
        for name in hg.nodes[ntype].data:
            local_data = node_feats[ntype + '/' + name][type_nid]
            local_data1 = hg.nodes[ntype].data[name][orig_type_nid]
            assert np.all(local_data.numpy() == local_data1.numpy())

    for etype_id in th.unique(subg_etype):
        etype = etypes[etype_id]
        idx = subg_etype == etype_id
        exist = hg[etype].has_edges_between(subg_src_id[idx], subg_dst_id[idx])
        assert np.all(exist.numpy())
        eid = hg[etype].edge_ids(subg_src_id[idx], subg_dst_id[idx])
        assert np.all(eid.numpy() == subg.edata['orig_id'][idx].numpy())

        # This is global IDs after reshuffle.
        eid = subg.edata[dgl.EID][idx]
        etype_ids1, type_eid = eid_map(eid)
        orig_type_eid = subg.edata['orig_id'][idx]
        # All edges should have the same edge type.
        assert np.all(etype_ids1.numpy() == int(etype_id))

        # Check edge data.
        for name in hg.edges[etype].data:
            local_data = edge_feats[etype + '/' + name][type_eid]
            local_data1 = hg.edges[etype].data[name][orig_type_eid]
            assert np.all(local_data.numpy() == local_data1.numpy())
