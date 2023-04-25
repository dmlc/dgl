import json

import dgl
import numpy as np
import torch as th
from ogb.nodeproppred import DglNodePropPredDataset

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
print(hg)

# OGB-MAG is stored in heterogeneous format. We need to convert it into homogeneous format.
g = dgl.to_homogeneous(hg)
g.ndata["orig_id"] = g.ndata[dgl.NID]
g.edata["orig_id"] = g.edata[dgl.EID]
print("|V|=" + str(g.num_nodes()))
print("|E|=" + str(g.num_edges()))
print("|NTYPE|=" + str(len(th.unique(g.ndata[dgl.NTYPE]))))

# Store the metadata of nodes.
num_node_weights = 0
node_data = [g.ndata[dgl.NTYPE].numpy()]
for ntype_id in th.unique(g.ndata[dgl.NTYPE]):
    node_data.append((g.ndata[dgl.NTYPE] == ntype_id).numpy())
    num_node_weights += 1
node_data.append(g.ndata["orig_id"].numpy())
node_data = np.stack(node_data, 1)
np.savetxt("mag_nodes.txt", node_data, fmt="%d", delimiter=" ")

# Store the node features
node_feats = {}
for ntype in hg.ntypes:
    for name in hg.nodes[ntype].data:
        node_feats[ntype + "/" + name] = hg.nodes[ntype].data[name]
dgl.data.utils.save_tensors("node_feat.dgl", node_feats)

# Store the metadata of edges.
# ParMETIS cannot handle duplicated edges and self-loops. We should remove them
# in the preprocessing.
src_id, dst_id = g.edges()
# Remove self-loops
self_loop_idx = src_id == dst_id
not_self_loop_idx = src_id != dst_id
self_loop_src_id = src_id[self_loop_idx]
self_loop_dst_id = dst_id[self_loop_idx]
self_loop_orig_id = g.edata["orig_id"][self_loop_idx]
self_loop_etype = g.edata[dgl.ETYPE][self_loop_idx]
src_id = src_id[not_self_loop_idx]
dst_id = dst_id[not_self_loop_idx]
orig_id = g.edata["orig_id"][not_self_loop_idx]
etype = g.edata[dgl.ETYPE][not_self_loop_idx]
# Remove duplicated edges.
ids = (src_id * g.num_nodes() + dst_id).numpy()
uniq_ids, idx = np.unique(ids, return_index=True)
duplicate_idx = np.setdiff1d(np.arange(len(ids)), idx)
duplicate_src_id = src_id[duplicate_idx]
duplicate_dst_id = dst_id[duplicate_idx]
duplicate_orig_id = orig_id[duplicate_idx]
duplicate_etype = etype[duplicate_idx]
src_id = src_id[idx]
dst_id = dst_id[idx]
orig_id = orig_id[idx]
etype = etype[idx]
edge_data = th.stack([src_id, dst_id, orig_id, etype], 1)
np.savetxt("mag_edges.txt", edge_data.numpy(), fmt="%d", delimiter=" ")
removed_edge_data = th.stack(
    [
        th.cat([self_loop_src_id, duplicate_src_id]),
        th.cat([self_loop_dst_id, duplicate_dst_id]),
        th.cat([self_loop_orig_id, duplicate_orig_id]),
        th.cat([self_loop_etype, duplicate_etype]),
    ],
    1,
)
np.savetxt(
    "mag_removed_edges.txt", removed_edge_data.numpy(), fmt="%d", delimiter=" "
)
print(
    "There are {} edges, remove {} self-loops and {} duplicated edges".format(
        g.num_edges(), len(self_loop_src_id), len(duplicate_src_id)
    )
)

# Store the edge features
edge_feats = {}
for etype in hg.etypes:
    for name in hg.edges[etype].data:
        edge_feats[etype + "/" + name] = hg.edges[etype].data[name]
dgl.data.utils.save_tensors("edge_feat.dgl", edge_feats)

# Store the basic metadata of the graph.
graph_stats = [g.num_nodes(), len(src_id), num_node_weights]
with open("mag_stats.txt", "w") as filehandle:
    filehandle.writelines(
        "{} {} {}".format(graph_stats[0], graph_stats[1], graph_stats[2])
    )

# Store the ID ranges of nodes and edges of the entire graph.
nid_ranges = {}
eid_ranges = {}
for ntype in hg.ntypes:
    ntype_id = hg.get_ntype_id(ntype)
    nid = th.nonzero(g.ndata[dgl.NTYPE] == ntype_id, as_tuple=True)[0]
    per_type_nid = g.ndata["orig_id"][nid]
    assert np.all((per_type_nid == th.arange(len(per_type_nid))).numpy())
    assert np.all((nid == th.arange(nid[0], nid[-1] + 1)).numpy())
    nid_ranges[ntype] = [int(nid[0]), int(nid[-1] + 1)]
for etype in hg.etypes:
    etype_id = hg.get_etype_id(etype)
    eid = th.nonzero(g.edata[dgl.ETYPE] == etype_id, as_tuple=True)[0]
    assert np.all((eid == th.arange(eid[0], eid[-1] + 1)).numpy())
    eid_ranges[etype] = [int(eid[0]), int(eid[-1] + 1)]
with open("mag.json", "w") as outfile:
    json.dump({"nid": nid_ranges, "eid": eid_ranges}, outfile, indent=4)
