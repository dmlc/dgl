import dgl
import dgl.graphbolt as gb
import numpy as np
import torch as th

# [TODO][P0] Set up distributed environment.

"""
num_trainers = 8
num_servers = 4
num_samplers = 0
part_config = ./ogbn-products.json
ip_config = ./ip_config.txt
"""

args = {}

# Initialize distributed environment
dgl.distributed.initialize(args.ip_config)
th.distributed.init_process_group(backend=args.backend)
# [TODO][P0] Convert dgl partitioned graphs to graphbolt.CSCSamplingGraph.
#           done@2023-10-23 16:49:00
#           see details in: https://github.com/Rhett-Ying/dgl/commits/gb_distdgl
#                           ddce1d42de016be040cd0f8a5e71f2a10148de82
'''
In [1]: part_config='/home/ubuntu/workspace/dgl_2/data/ogbn-mag.json'
In [3]: dgl.distributed.convert_dgl_partition_to_csc_sampling_graph(part_config, store_orig_nids=True)
In [7]: !ls data/part0 -lh
total 1.1G
-rw-rw-r-- 1 ubuntu ubuntu 207M Oct 23 08:44 csc_sampling_graph.tar
-rw-rw-r-- 1 ubuntu ubuntu 694M Oct 23 02:47 graph.dgl

In [8]: !ls data/part1 -lh
total 1.1G
-rw-rw-r-- 1 ubuntu ubuntu 202M Oct 23 08:44 csc_sampling_graph.tar
-rw-rw-r-- 1 ubuntu ubuntu 678M Oct 23 02:47 graph.dgl
'''

# [TODO][P0] Load `CSCSamplingGraph` into `DistGraph`.
#           done@2023-10-24 15:10:00
#           see details in: https://github.com/Rhett-Ying/dgl/commits/gb_distdgl
#                           222dd2bd51084cc4f242148b0a7e6e5d91e0ae80
## NID/EIDs are required.
g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)

# Generate train/val/test splits
##############
# train/val/test splits could be generated offline, then `train/val/test_masks`
#   could be offloaded.
# No change is required as `node_split` requires graph parition book and
#   masks only.
# This should be part of `OnDiskDataset::TVT`.
# [TODO][P1]: Add a standalone API to generate train/val/test splits.
##############
gpb = g.get_partition_book()
train_nids = dgl.distributed.node_split(g.ndata["train_masks"], gpb)
val_nids = dgl.distributed.node_split(g.ndata["val_masks"], gpb)
test_nids = dgl.distributed.node_split(g.ndata["test_masks"], gpb)
all_nids = dgl.distributed.node_split(th.arange(g.num_nodes()), gpb)

# [TODO][P2] How to handle feature data such as 'feat', 'mask'?
# Just use `g.ndata['feat']` for now. As no more memory could be offloaded.
# GB: feat_data = gb.OnDiskDataset().feature
# DistDGL: feat_data = g.ndata['feat'] # DistTensor


# Train.
##############
# GraphBolt version
# [TODO][P0] Add `gb.distributed_sample_neighbor` API.
# [TODO][P0] `remote_sample_neighbor()` returns original global node pairs + eids.
# [TODO][P0] Upldate `dgl.distributed.merge_graphs` API.
#     https://github.com/dmlc/dgl/blob/7439b7e73bdb85b4285ab01f704ac5a4f77c927e/python/dgl/distributed/graph_services.py#L440.
##############
"""
datapipe = gb.ItemSampler(item_set, batch_size=batch_size, shuffle=shuffle)
datapipe = datapipe.sample_neighbor(g._graph, fanouts=fanouts)
datapipe = datapipe.to_dgl()
device = th.device("cpu")
datapipe = datapipe.copy_to(device)
data_loader = gb.MultiProcessDataLoader(datapipe, num_workers=num_workers)
"""
sampler = dgl.dataloading.NeighborSampler([25, 10])
train_dataloader = dgl.distributed.DistDataLoader(
    g, train_nids, sampler=sampler, batch_size=args.batch_size, shuffle=True
)
model = None
for mini_batch in train_dataloader:
    in_feats = g.ndata["feat"][mini_batch.input_nodes]
    labels = g.ndata["label"][mini_batch.output_nodes]
    _ = model(mini_batch, in_feats)

# Evaluate.
model.eval()
sampler = dgl.dataloading.NeighborSampler([-1])
val_dataloader = dgl.distributed.DistDataLoader(
    g, val_nids, sampler=sampler, batch_size=args.batch_size, shuffle=False
)
test_dataloader = dgl.distributed.DistDataLoader(
    g, test_nids, sampler=sampler, batch_size=args.batch_size, shuffle=False
)