import torch as th
import dgl
import torch.nn as nn
from dgl.nn.functional import edge_softmax
import time
import numpy as np
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

data = CoraGraphDataset()
g = data[0]
n_edges = data.graph.number_of_edges()
values = th.randn((n_edges,50)).requires_grad_(True)
g.edata['e'] = values
edata = g.edata['e']

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,g, ndata):
        return edge_softmax(g, ndata)
loss_fcn = nn.CrossEntropyLoss()
m = Model()
a = m(g, edata)
loss = loss_fcn(a, edata)
loss.backward()