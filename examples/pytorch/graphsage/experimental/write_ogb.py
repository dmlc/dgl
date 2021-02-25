import dgl
import json
import torch as th
import numpy as np
import argparse
from ogb.nodeproppred import DglNodePropPredDataset
from pyinstrument import Profiler

argparser = argparse.ArgumentParser("Partition builtin graphs")
argparser.add_argument('--dataset', type=str, default='ogbn-products',
                       help='datasets: ogbn-products, ogbn-papers100M')
args = argparser.parse_args()

data = DglNodePropPredDataset(name=args.dataset)
splitted_idx = data.get_idx_split()
g, labels = data[0]
g.ndata['labels'] = labels[:, 0]
print(g.ndata['labels'].dtype)

# Find the node IDs in the training, validation, and test set.
train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
train_mask = th.zeros((g.number_of_nodes(),), dtype=th.bool)
train_mask[train_nid] = True
val_mask = th.zeros((g.number_of_nodes(),), dtype=th.bool)
val_mask[val_nid] = True
test_mask = th.zeros((g.number_of_nodes(),), dtype=th.bool)
test_mask[test_nid] = True
g.ndata['train_mask'] = train_mask
g.ndata['val_mask'] = val_mask
g.ndata['test_mask'] = test_mask
print('|V|=' + str(g.number_of_nodes()))
print('|E|=' + str(g.number_of_edges()))
print(g)

profiler = Profiler()
profiler.start()

# Store the metadata of nodes.
# All nodes have the same node type.
node_data = [th.zeros(g.number_of_nodes(), dtype=th.int64)]
# All nodes have the same weight.
# TODO(zhengda) do we have to have the node weigths?
node_data.append(th.ones(g.number_of_nodes(), dtype=th.int64))
num_node_weights = 1
node_data.append(th.arange(g.number_of_nodes()))
node_data = th.stack(node_data, dim=1)
np.savetxt(args.dataset + '_nodes.txt', node_data.numpy(), fmt='%d', delimiter=' ')

# Store the node features
node_feats = {}
assert len(g.ntypes) == 1
for ntype in g.ntypes:
    print(ntype)
    for name in g.nodes[ntype].data:
        node_feats[ntype + '/' + name] = g.nodes[ntype].data[name]
dgl.data.utils.save_tensors("node_feat.dgl", node_feats)

# Store the metadata of edges.
src_id, dst_id = g.edges()
no_self_loop = src_id != dst_id
src_id1 = src_id[no_self_loop].long()
dst_id1 = dst_id[no_self_loop].long()
uniq_edges = th.unique(src_id1 * g.number_of_nodes() + dst_id1)
num_edges = len(src_id1)
print('#self loops:', g.number_of_edges() - num_edges)
assert len(src_id1) == len(uniq_edges), \
        'There are duplicated edges. METIS does not allow duplicated edges'
edge_data = th.stack([src_id1, dst_id1,
                      th.arange(num_edges),
                      th.zeros(num_edges, dtype=th.int64)], 1)
np.savetxt(args.dataset + '_edges.txt', edge_data.numpy(), fmt='%d', delimiter=' ')

# Store the edge features
edge_feats = {}
assert len(g.etypes) == 1
for etype in g.etypes:
    for name in g.edges[etype].data:
        edge_feats[etype + '/' + name] = g.edges[etype].data[name][no_self_loop]
dgl.data.utils.save_tensors("edge_feat.dgl", edge_feats)

# Store the basic metadata of the graph.
graph_stats = [g.number_of_nodes(), num_edges, 1]
with open(args.dataset + '_stats.txt', 'w') as filehandle:
    filehandle.writelines("{} {} {}".format(graph_stats[0], graph_stats[1], graph_stats[2]))

# Store the ID ranges of nodes and edges of the entire graph.
nid_ranges = {ntype: [0, g.number_of_nodes()]}
eid_ranges = {etype: [0, num_edges]}
with open(args.dataset + '.json', 'w') as outfile:
    json.dump({'nid': nid_ranges, 'eid': eid_ranges}, outfile, indent=4)

profiler.stop()
print(profiler.output_text(unicode=True, color=True))
