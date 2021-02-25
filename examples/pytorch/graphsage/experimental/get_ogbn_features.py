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
argparser.add_argument('--output', type=str, required=True,
                       help='the output directory that stores the partition results.')
args = argparser.parse_args()

dataset = DglNodePropPredDataset(name=args.dataset)
g, labels = dataset[0]

split_idx = dataset.get_idx_split()
train_idx = split_idx["train"]
val_idx = split_idx["valid"]
test_idx = split_idx["test"]
labels = labels[:, 0]

train_mask = th.zeros((g.number_of_nodes(),), dtype=th.bool)
train_mask[train_idx] = True
val_mask = th.zeros((g.number_of_nodes(),), dtype=th.bool)
val_mask[val_idx] = True
test_mask = th.zeros((g.number_of_nodes(),), dtype=th.bool)
test_mask[test_idx] = True
g.ndata['train_mask'] = train_mask
g.ndata['val_mask'] = val_mask
g.ndata['test_mask'] = test_mask
g.ndata['labels'] = th.tensor(labels)
g.ndata['features'] = g.ndata['feat']
del g.ndata['feat']

meta_file = args.output + '/' + args.dataset + '.json'
with open(meta_file) as json_file:
    metadata = json.load(json_file)

for part_id in range(metadata['num_parts']):
    subg = dgl.load_graphs(args.output + '/part{}/graph.dgl'.format(part_id))[0][0]

    node_data = {}
    for ntype in g.ntypes:
        local_node_idx = th.logical_and(subg.ndata['inner_node'].bool(),
                                        subg.ndata[dgl.NTYPE] == g.get_ntype_id(ntype))
        local_nodes = subg.ndata['orig_id'][local_node_idx].numpy()
        for name in g.nodes[ntype].data:
            node_data[ntype + '/' + name] = g.nodes[ntype].data[name][local_nodes]
    print('node features:', node_data.keys())
    dgl.data.utils.save_tensors(metadata['part-{}'.format(part_id)]['node_feats'], node_data)

    edge_data = {}
    for etype in g.etypes:
        local_edges = subg.edata['orig_id'][subg.edata[dgl.ETYPE] == g.get_etype_id(etype)]
        for name in g.edges[etype].data:
            edge_data[etype + '/' + name] = g.edges[etype].data[name][local_edges]
    print('edge features:', edge_data.keys())
    dgl.data.utils.save_tensors(metadata['part-{}'.format(part_id)]['edge_feats'], edge_data)
