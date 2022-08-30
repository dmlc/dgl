import dgl
from dgl import rand_graph
import dgl.ops as OPS
from dgl.cuda import to_dgl_stream_handle
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
    with dgl.cuda.stream(s):
        gg = g.to(device=F.ctx())
    s.synchronize()
    OPS.copy_u_sum(gg, xx)

    # launch on new stream created via torch.cuda
    s = torch.cuda.Stream(device=F.ctx())
    with torch.cuda.stream(s):
        xx = x.to(device=F.ctx(), non_blocking=True)
    with dgl.cuda.stream(s):
        gg = g.to(device=F.ctx())
    s.synchronize()
    OPS.copy_u_sum(gg, xx)

    # launch on default stream used in DGL
    xx = x.to(device=F.ctx())
    gg = g.to(device=F.ctx())
    OPS.copy_u_sum(gg, xx)

@unittest.skipIf(F._default_context_str == 'cpu', reason="stream only runs on GPU.")
def test_set_get_stream():
    s = torch.cuda.Stream(device=F.ctx())
    dgl.cuda.set_stream(s)
    assert to_dgl_stream_handle(s).value == dgl.cuda.current_stream(F.ctx()).value

if __name__ == '__main__':
    test_basics()
    test_set_get_stream()
