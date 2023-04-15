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

split_idx = dataset.get_idx_split()
train_idx = split_idx["train"]["paper"]
val_idx = split_idx["valid"]["paper"]
test_idx = split_idx["test"]["paper"]
paper_labels = labels["paper"].squeeze()

train_mask = th.zeros((hg.num_nodes("paper"),), dtype=th.bool)
train_mask[train_idx] = True
val_mask = th.zeros((hg.num_nodes("paper"),), dtype=th.bool)
val_mask[val_idx] = True
test_mask = th.zeros((hg.num_nodes("paper"),), dtype=th.bool)
test_mask[test_idx] = True
hg.nodes["paper"].data["train_mask"] = train_mask
hg.nodes["paper"].data["val_mask"] = val_mask
hg.nodes["paper"].data["test_mask"] = test_mask
hg.nodes["paper"].data["labels"] = paper_labels

with open("outputs/mag.json") as json_file:
    metadata = json.load(json_file)

for part_id in range(metadata["num_parts"]):
    subg = dgl.load_graphs("outputs/part{}/graph.dgl".format(part_id))[0][0]

    node_data = {}
    for ntype in hg.ntypes:
        local_node_idx = th.logical_and(
            subg.ndata["inner_node"].bool(),
            subg.ndata[dgl.NTYPE] == hg.get_ntype_id(ntype),
        )
        local_nodes = subg.ndata["orig_id"][local_node_idx].numpy()
        for name in hg.nodes[ntype].data:
            node_data[ntype + "/" + name] = hg.nodes[ntype].data[name][
                local_nodes
            ]
    print("node features:", node_data.keys())
    dgl.data.utils.save_tensors(
        "outputs/" + metadata["part-{}".format(part_id)]["node_feats"],
        node_data,
    )

    edge_data = {}
    for etype in hg.etypes:
        local_edges = subg.edata["orig_id"][
            subg.edata[dgl.ETYPE] == hg.get_etype_id(etype)
        ]
        for name in hg.edges[etype].data:
            edge_data[etype + "/" + name] = hg.edges[etype].data[name][
                local_edges
            ]
    print("edge features:", edge_data.keys())
    dgl.data.utils.save_tensors(
        "outputs/" + metadata["part-{}".format(part_id)]["edge_feats"],
        edge_data,
    )
