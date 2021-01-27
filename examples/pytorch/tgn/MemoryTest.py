# TODO: Use lice of wikipedia dataset to test the functionality of memory module
from dgl.function.message import copy_e
import torch
import torch.nn as nn
import dgl
from dgl import function as fn
import numpy as np
from modules import MemoryModule, MemoryOperation

# Load wikipedia dataset

g,_ = dgl.load_graphs('wikipedia.bin')

g = g[0]

src,dst = g.edges()
g.edata['timestamp'] = g.edata['timestamp'].float()

# Create memory module coresponding to it

memory = MemoryModule(g.num_nodes(),500)

memory_ops = MemoryOperation(updater_type='gru',
                             memory = memory,
                             feat_dim=g.edata['feats'].shape[1],
                             temporal_dim=20)

# Slice the graph, assume NID preserved

p_subg = dgl.edge_subgraph(g,range(200))

c_subg = dgl.edge_subgraph(g,range(200,400))

# Only operate on c_subg

ret = memory_ops(c_subg)
final = dgl.sum_nodes(ret,'s')
loss = torch.norm(final,p=2)
loss.backward()

print("Summary",ret)



