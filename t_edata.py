import dgl
from dgl.edata_fixture import use_edata_for_update
import dgl.function as fn

from dgl.data import CoraGraphDataset

ds = CoraGraphDataset()

g = ds[0]
g.update_all(fn.copy_u('feat', 'h'), fn.sum('h', 'out'))

import torch as th

g.edata['ee'] = th.randn(g.num_edges(), requires_grad=True)

with use_edata_for_update("ee"):
    g.update_all(fn.copy_u('feat', 'h'), fn.sum('h', 'out'))

loss = g.ndata['out'].sum()
loss.backward()
print(g.edata['ee'].grad)