import dgl
import networkx as nx
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

def capsule_message(src, edge):
    return {'ft' : src['ft'], 'bij' : edge['b']}

class GATReduce(nn.Module):
    def __init__(self, attn_drop):
        super(GATReduce, self).__init__()
        self.attn_drop = attn_drop

    def forward(self, node, msgs):
        a = torch.unsqueeze(node['a'], 0)  # shape (1, 1)
        ft = torch.cat([torch.unsqueeze(m['ft'], 0) for m in msgs], dim=0) # shape (deg, D)
        # attention
        e = F.softmax(a, dim=0)
        if self.attn_drop != 0.0:
            e = F.dropout(e, self.attn_drop)
        return torch.sum(e * ft, dim=0) # shape (D,)

class Capsule(nn.Module):
    def __init__(self):
        super(Capsule, self).__init__()
        self.g = dgl.DGLGraph(nx.from_numpy_matrix(np.ones((10, 10))))

    def forward(self, node, msgs):
        a1 = torch.unsqueeze(node['a1'], 0)  # shape (1, 1)
        a2 = torch.cat([torch.unsqueeze(m['a2'], 0) for m in msgs], dim=0)  # shape (deg, 1)
        ft = torch.cat([torch.unsqueeze(m['ft'], 0) for m in msgs], dim=0)  # shape (deg, D)

