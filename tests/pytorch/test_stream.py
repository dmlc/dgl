from statistics import mean
import unittest
import numpy as np
import torch
import dgl
import dgl.ndarray as nd
from dgl import rand_graph
import dgl.ops as OPS
from dgl._ffi.streams import to_dgl_stream_handle, _dgl_get_stream
from dgl.utils import to_dgl_context
import backend as F

# borrowed from PyTorch, torch/testing/_internal/common_utils.py
def _get_cycles_per_ms() -> float:
    """Measure and return approximate number of cycles per millisecond for torch.cuda._sleep
    """

    def measure() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda._sleep(1000000)
        end.record()
        end.synchronize()
        cycles_per_ms = 1000000 / start.elapsed_time(end)
        return cycles_per_ms

    # Get 10 values and remove the 2 max and 2 min and return the avg.
    # This is to avoid system disturbance that skew the results, e.g.
    # the very first cuda call likely does a bunch of init, which takes
    # much longer than subsequent calls.
    num = 10
    vals = []
    for _ in range(num):
        vals.append(measure())
    vals = sorted(vals)
    return mean(vals[2 : num - 2])

@unittest.skipIf(F._default_context_str == 'cpu', reason="stream only runs on GPU.")
def test_basics():
    g = rand_graph(10, 20, device=F.cpu())
    x = torch.ones(g.num_nodes(), 10)

    # launch on default stream fetched via torch.cuda
    s = torch.cuda.default_stream(device=F.ctx())
    with torch.cuda.stream(s):
        xx = x.to(device=F.ctx(), non_blocking=True)
        gg = g.to(device=F.ctx())
    s.synchronize()
    OPS.copy_u_sum(gg, xx)

    # launch on new stream created via torch.cuda
    s = torch.cuda.Stream(device=F.ctx())
    with torch.cuda.stream(s):
        xx = x.to(device=F.ctx(), non_blocking=True)
        gg = g.to(device=F.ctx())
    s.synchronize()
    OPS.copy_u_sum(gg, xx)

    # launch on default stream used in DGL
    xx = x.to(device=F.ctx())
    gg = g.to(device=F.ctx())
    OPS.copy_u_sum(gg, xx)

@unittest.skipIf(F._default_context_str == 'cpu', reason="stream only runs on GPU.")
def test_set_get_stream():
    current_stream = torch.cuda.current_stream()
    # test setting another stream
    s = torch.cuda.Stream(device=F.ctx())
    torch.cuda.set_stream(s)
    assert to_dgl_stream_handle(s).value == _dgl_get_stream(to_dgl_context(F.ctx())).value
    # revert to default stream
    torch.cuda.set_stream(current_stream)

@unittest.skipIf(F._default_context_str == 'cpu', reason="stream only runs on GPU.")
def test_record_stream_smoke():
    # smoke test to ensure it can work
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        t = nd.array(np.array([1., 2., 3., 4.]), ctx=nd.gpu(0))
        g = rand_graph(10, 20, device=F.cuda())
    # NDArray is an internal object, just use DGLStreamHandle
    t.record_stream(to_dgl_stream_handle(torch.cuda.current_stream()))
    g.record_stream(torch.cuda.current_stream())

@unittest.skipIf(F._default_context_str == 'cpu', reason="stream only runs on GPU.")
# borrowed from PyTorch, test/test_cuda.py: test_record_stream()
# TODO(Xin): Currently we don't have an equivalent API for dgl.graph
# as tensor.data_ptr() to check if the memory has been reused or not.
# The underlying sparse matrices or NDArrays of dgl.graph have not 
# been exposed to Python and we have not way to access them.
# Therefore, we only test record_stream for NDArray here.
def test_record_stream():
    cycles_per_ms = _get_cycles_per_ms()

    t = nd.array(np.array([1., 2., 3., 4.], dtype=np.float32), ctx=nd.cpu())
    t.pin_memory_()
    result = nd.empty([4], ctx=nd.gpu(0))
    stream = torch.cuda.Stream()
    ptr = [None]

    # Performs the CPU->GPU copy in a background stream
    def perform_copy():
        with torch.cuda.stream(stream):
            tmp = t.copyto(nd.gpu(0))
            ptr[0] = F.from_dgl_nd(tmp).data_ptr()
        torch.cuda.current_stream().wait_stream(stream)
        tmp.record_stream(
            to_dgl_stream_handle(torch.cuda.current_stream()))
        torch.cuda._sleep(int(50 * cycles_per_ms))  # delay the copy
        result.copyfrom(tmp)

    perform_copy()
    with torch.cuda.stream(stream):
        tmp2 = nd.empty([4], ctx=nd.gpu(0))
        assert F.from_dgl_nd(tmp2).data_ptr() != ptr[0], 'allocation re-used to soon'

    assert torch.equal(F.from_dgl_nd(result).cpu(), torch.tensor([1., 2., 3., 4.]))

    # Check that the block will be re-used after the main stream finishes
    torch.cuda.current_stream().synchronize()
    with torch.cuda.stream(stream):
        tmp3 = nd.empty([4], ctx=nd.gpu(0))
        assert F.from_dgl_nd(tmp3).data_ptr() == ptr[0], 'allocation not re-used'

if __name__ == '__main__':
    test_basics()
    test_set_get_stream()
    test_record_stream_smoke()
    test_record_stream()
