import dgl
import torch as th
import torch.nn as nn
from DGLRoutingLayer import DGLRoutingLayer
from torch.nn import functional as F

g = dgl.DGLGraph()
g.graph_data = {}

in_nodes = 20
out_nodes = 10
g.graph_data["in_nodes"] = in_nodes
g.graph_data["out_nodes"] = out_nodes
all_nodes = in_nodes + out_nodes
g.add_nodes(all_nodes)


in_indx = list(range(in_nodes))
out_indx = list(range(in_nodes, in_nodes + out_nodes))
g.graph_data["in_indx"] = in_indx
g.graph_data["out_indx"] = out_indx

# add edges use edge broadcasting
for u in out_indx:
    g.add_edges(in_indx, u)
# init states
f_size = 4
g.ndata["v"] = th.zeros(all_nodes, f_size)
g.edata["u_hat"] = th.randn(in_nodes * out_nodes, f_size)
g.edata["b"] = th.randn(in_nodes * out_nodes, 1)

routing_layer = DGLRoutingLayer(g)

entropy_list = []
for i in range(15):
    routing_layer()
    dist_matrix = g.edata["c"].view(in_nodes, out_nodes)
    entropy = (-dist_matrix * th.log(dist_matrix)).sum(dim=0)
    entropy_list.append(entropy.data.numpy())
    std = dist_matrix.std(dim=0)
