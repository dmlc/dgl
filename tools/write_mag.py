import dgl
import json
import torch as th
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
from pyinstrument import Profiler

dataset = DglNodePropPredDataset(name='ogbn-mag')
hg, labels = dataset[0]
#subg_nodes = {}
#for ntype in hg.ntypes:
#    subg_nodes[ntype] = np.random.choice(hg.number_of_nodes(ntype), int(hg.number_of_nodes(ntype) / 5), replace=False)
#hg = dgl.compact_graphs(dgl.node_subgraph(hg, subg_nodes))

profiler = Profiler()
profiler.start()

g = dgl.to_homogeneous(hg)
g.ndata['orig_id'] = g.ndata[dgl.NID]
g.edata['orig_id'] = g.edata[dgl.EID]
g.ndata['feat'] = th.arange(g.number_of_nodes())
g.edata['feat'] = th.arange(g.number_of_edges())
dgl.save_graphs('mag.dgl', [g])
print('|V|=' + str(g.number_of_nodes()))
print('|E|=' + str(g.number_of_edges()))
print('|NTYPE|=' + str(len(th.unique(g.ndata[dgl.NTYPE]))))

node_data = th.stack([g.ndata[dgl.NTYPE],
                      g.ndata[dgl.NTYPE] == 0,
                      g.ndata[dgl.NTYPE] == 1,
                      g.ndata['orig_id'],
                      g.ndata['feat']], 1)
np.savetxt('mag_nodes.txt', node_data.numpy(), fmt='%d', delimiter=' ')

src_id, dst_id = g.edges()
edge_data = th.stack([src_id, dst_id,
                      g.edata['orig_id'],
                      g.edata[dgl.ETYPE],
                      g.edata['feat']], 1)
np.savetxt('mag_edges.txt', edge_data.numpy(), fmt='%d', delimiter=' ')

graph_stats = [g.number_of_nodes(), g.number_of_edges(), 2]
with open('mag_stats.txt', 'w') as filehandle:
    filehandle.writelines("{} {} {}".format(graph_stats[0], graph_stats[1], graph_stats[2]))

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

profiler.stop()
print(profiler.output_text(unicode=True, color=True))
