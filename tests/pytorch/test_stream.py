from dgl import rand_graph
import dgl._ffi.streams as FS
import dgl.ops as OPS
import unittest
import backend as F
import torch


@unittest.skipIf(F._default_context_str == 'cpu', reason="stream only runs on GPU.")
def test_basics():
    g = rand_graph(10, 20, device=F.cpu())
    x = torch.ones(g.num_nodes(), 10)

    # launch on default stream fetched via torch.cuda
    s = torch.cuda.default_stream(device=F.ctx())
    with torch.cuda.stream(s):
        xx = x.to(device=F.ctx(), non_blocking=True)
    with FS.stream(s):
        gg = g.to(device=F.ctx())
    s.synchronize()
    OPS.copy_u_sum(gg, xx)

    # launch on new stream created via torch.cuda
    s = torch.cuda.Stream(device=F.ctx())
    with torch.cuda.stream(s):
        xx = x.to(device=F.ctx(), non_blocking=True)
    with FS.stream(s):
        gg = g.to(device=F.ctx())
    s.synchronize()
    OPS.copy_u_sum(gg, xx)

    # launch on default stream used in DGL
    xx = x.to(device=F.ctx())
    gg = g.to(device=F.ctx())
    OPS.copy_u_sum(gg, xx)


if __name__ == '__main__':
    test_basics()
