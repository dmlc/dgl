import torch
import dgl
import dgl.backend as F

g = dgl.rand_graph(10, 15).int().to(torch.device(0))
gidx = g._graph
u = torch.rand((10,2,8), device=torch.device(0))
v = torch.rand((10,2,8), device=torch.device(0))
e = torch.zeros((15,2,1), device=torch.device(0))
u = F.zerocopy_to_dgl_ndarray(u)
v = F.zerocopy_to_dgl_ndarray(v)
e = F.zerocopy_to_dgl_ndarray_for_write(e)
dgl.sparse._CAPI_FG_SDDMMTreeReduction(gidx, u, v, e)
print(e)