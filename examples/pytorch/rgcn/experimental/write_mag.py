import dgl
import json
import torch as th
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset

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
print(hg)
#subg_nodes = {}
#for ntype in hg.ntypes:
#    subg_nodes[ntype] = np.random.choice(hg.number_of_nodes(ntype), int(hg.number_of_nodes(ntype) / 5), replace=False)
#hg = dgl.compact_graphs(dgl.node_subgraph(hg, subg_nodes))

# OGB-MAG is stored in heterogeneous format. We need to convert it into homogeneous format.
g = dgl.to_homogeneous(hg)
g.ndata['orig_id'] = g.ndata[dgl.NID]
g.edata['orig_id'] = g.edata[dgl.EID]
print('|V|=' + str(g.number_of_nodes()))
print('|E|=' + str(g.number_of_edges()))
print('|NTYPE|=' + str(len(th.unique(g.ndata[dgl.NTYPE]))))

# Store the metadata of nodes.
num_node_weights = 0
node_data = [g.ndata[dgl.NTYPE].numpy()]
for ntype_id in th.unique(g.ndata[dgl.NTYPE]):
    node_data.append((g.ndata[dgl.NTYPE] == ntype_id).numpy())
    num_node_weights += 1
node_data.append(g.ndata['orig_id'].numpy())
node_data = np.stack(node_data, 1)
np.savetxt('mag_nodes.txt', node_data, fmt='%d', delimiter=' ')

# Store the node features
node_feats = {}
for ntype in hg.ntypes:
    for name in hg.nodes[ntype].data:
        node_feats[ntype + '/' + name] = hg.nodes[ntype].data[name]
dgl.data.utils.save_tensors("node_feat.dgl", node_feats)

# Store the metadata of edges.
src_id, dst_id = g.edges()
edge_data = th.stack([src_id, dst_id,
                      g.edata['orig_id'],
                      g.edata[dgl.ETYPE]], 1)
np.savetxt('mag_edges.txt', edge_data.numpy(), fmt='%d', delimiter=' ')

# Store the edge features
edge_feats = {}
for etype in hg.etypes:
    for name in hg.edges[etype].data:
        edge_feats[etype + '/' + name] = hg.edges[etype].data[name]
dgl.data.utils.save_tensors("edge_feat.dgl", edge_feats)

# Store the basic metadata of the graph.
graph_stats = [g.number_of_nodes(), g.number_of_edges(), num_node_weights]
with open('mag_stats.txt', 'w') as filehandle:
    filehandle.writelines("{} {} {}".format(graph_stats[0], graph_stats[1], graph_stats[2]))

# Store the ID ranges of nodes and edges of the entire graph.
nid_ranges = {}
eid_ranges = {}
for ntype in hg.ntypes:
    ntype_id = hg.get_ntype_id(ntype)
    nid = th.nonzero(g.ndata[dgl.NTYPE] == ntype_id, as_tuple=True)[0]
    per_type_nid = g.ndata['orig_id'][nid]
    assert np.all((per_type_nid == th.arange(len(per_type_nid))).numpy())
    assert np.all((nid == th.arange(nid[0], nid[-1] + 1)).numpy())
    nid_ranges[ntype] = [int(nid[0]), int(nid[-1] + 1)]
for etype in hg.etypes:
    etype_id = hg.get_etype_id(etype)
    eid = th.nonzero(g.edata[dgl.ETYPE] == etype_id, as_tuple=True)[0]
    assert np.all((eid == th.arange(eid[0], eid[-1] + 1)).numpy())
    eid_ranges[etype] = [int(eid[0]), int(eid[-1] + 1)]
with open('mag.json', 'w') as outfile:
    json.dump({'nid': nid_ranges, 'eid': eid_ranges}, outfile, indent=4)
