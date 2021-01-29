import dgl
import json
import torch as th
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset

dataset = DglNodePropPredDataset(name='ogbn-mag')
hg, labels = dataset[0]
subg_nodes = {}
for ntype in hg.ntypes:
    subg_nodes[ntype] = np.random.choice(hg.number_of_nodes(ntype), int(hg.number_of_nodes(ntype) / 5), replace=False)
subhg = dgl.compact_graphs(dgl.node_subgraph(hg, subg_nodes))

subg = dgl.to_homogeneous(subhg)
subg.ndata['orig_id'] = subg.ndata[dgl.NID]
subg.edata['orig_id'] = subg.edata[dgl.EID]
subg.ndata['feat'] = th.arange(subg.number_of_nodes())
subg.edata['feat'] = th.arange(subg.number_of_edges())
dgl.save_graphs('mag.dgl', [subg])
print('|V|=' + str(subg.number_of_nodes()))
print('|E|=' + str(subg.number_of_edges()))
print('|NTYPE|=' + str(len(th.unique(subg.ndata[dgl.NTYPE]))))

node_data = th.stack([subg.ndata[dgl.NTYPE],
                      subg.ndata[dgl.NTYPE] == 0,
                      subg.ndata[dgl.NTYPE] == 1,
                      subg.ndata['orig_id'],
                      subg.ndata['feat']], 1)
np.savetxt('mag_nodes.txt', node_data.numpy(), fmt='%d', delimiter=' ')

src_id, dst_id = subg.edges()
edge_data = th.stack([src_id, dst_id,
                      subg.edata['orig_id'],
                      subg.edata[dgl.ETYPE],
                      subg.edata['feat']], 1)
np.savetxt('mag_edges.txt', edge_data.numpy(), fmt='%d', delimiter=' ')

graph_stats = [subg.number_of_nodes(), subg.number_of_edges(), 2]
with open('mag_stats.txt', 'w') as filehandle:
    filehandle.writelines("{} {} {}".format(graph_stats[0], graph_stats[1], graph_stats[2]))

nid_ranges = {}
eid_ranges = {}
for ntype in subhg.ntypes:
    ntype_id = subhg.get_ntype_id(ntype)
    nid = th.nonzero(subg.ndata[dgl.NTYPE] == ntype_id, as_tuple=True)[0]
    per_type_nid = subg.ndata['orig_id'][nid]
    assert np.all((per_type_nid == th.arange(len(per_type_nid))).numpy())
    assert np.all((nid == th.arange(nid[0], nid[-1] + 1)).numpy())
    nid_ranges[ntype] = [int(nid[0]), int(nid[-1] + 1)]

for etype in subhg.etypes:
    etype_id = subhg.get_etype_id(etype)
    eid = th.nonzero(subg.edata[dgl.ETYPE] == etype_id, as_tuple=True)[0]
    assert np.all((eid == th.arange(eid[0], eid[-1] + 1)).numpy())
    eid_ranges[etype] = [int(eid[0]), int(eid[-1] + 1)]

with open('mag.json', 'w') as outfile:
    json.dump({'nid': nid_ranges, 'eid': eid_ranges}, outfile, indent=4)
