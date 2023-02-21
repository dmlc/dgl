import unittest
from statistics import mean

import backend as F

import dgl
import dgl.ndarray as nd
import dgl.ops as OPS
import numpy as np
import torch
from dgl import rand_graph
from dgl._ffi.streams import _dgl_get_stream, to_dgl_stream_handle
from dgl.utils import to_dgl_context


# borrowed from PyTorch, torch/testing/_internal/common_utils.py
def _get_cycles_per_ms() -> float:
    """Measure and return approximate number of cycles per millisecond for torch.cuda._sleep"""

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


@unittest.skipIf(
    F._default_context_str == "cpu", reason="stream only runs on GPU."
)
def test_basics():
    g = rand_graph(10, 20, device=F.cpu())
    x = torch.ones(g.num_nodes(), 10)
    result = OPS.copy_u_sum(g, x).to(F.ctx())

    # launch on default stream used in DGL
    xx = x.to(device=F.ctx())
    gg = g.to(device=F.ctx())
    OPS.copy_u_sum(gg, xx)
    assert torch.equal(OPS.copy_u_sum(gg, xx), result)

    # launch on new stream created via torch.cuda
    s = torch.cuda.Stream(device=F.ctx())
    with torch.cuda.stream(s):
        xx = x.to(device=F.ctx(), non_blocking=True)
        gg = g.to(device=F.ctx())
        OPS.copy_u_sum(gg, xx)
    s.synchronize()
    assert torch.equal(OPS.copy_u_sum(gg, xx), result)


@unittest.skipIf(
    F._default_context_str == "cpu", reason="stream only runs on GPU."
)
def test_set_get_stream():
    current_stream = torch.cuda.current_stream()
    # test setting another stream
    s = torch.cuda.Stream(device=F.ctx())
    torch.cuda.set_stream(s)
    assert (
        to_dgl_stream_handle(s).value
        == _dgl_get_stream(to_dgl_context(F.ctx())).value
    )
    # revert to default stream
    torch.cuda.set_stream(current_stream)


@unittest.skipIf(
    F._default_context_str == "cpu", reason="stream only runs on GPU."
)
# borrowed from PyTorch, test/test_cuda.py: test_record_stream()
def test_record_stream_ndarray():
    cycles_per_ms = _get_cycles_per_ms()

    t = nd.array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), ctx=nd.cpu())
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
        tmp.record_stream(to_dgl_stream_handle(torch.cuda.current_stream()))
        torch.cuda._sleep(int(50 * cycles_per_ms))  # delay the copy
        result.copyfrom(tmp)

    perform_copy()
    with torch.cuda.stream(stream):
        tmp2 = nd.empty([4], ctx=nd.gpu(0))
        assert (
            F.from_dgl_nd(tmp2).data_ptr() != ptr[0]
        ), "allocation re-used too soon"

    assert torch.equal(
        F.from_dgl_nd(result).cpu(), torch.tensor([1.0, 2.0, 3.0, 4.0])
    )

    # Check that the block will be re-used after the main stream finishes
    torch.cuda.current_stream().synchronize()
    with torch.cuda.stream(stream):
        tmp3 = nd.empty([4], ctx=nd.gpu(0))
        assert (
            F.from_dgl_nd(tmp3).data_ptr() == ptr[0]
        ), "allocation not re-used"


@unittest.skipIf(
    F._default_context_str == "cpu", reason="stream only runs on GPU."
)
def test_record_stream_graph_positive():
    cycles_per_ms = _get_cycles_per_ms()

    g = rand_graph(10, 20, device=F.cpu())
    g.create_formats_()
    x = torch.ones(g.num_nodes(), 10).to(F.ctx())
    g1 = g.to(F.ctx())
    # this is necessary to initialize the cusparse handle
    result = OPS.copy_u_sum(g1, x)
    torch.cuda.current_stream().synchronize()

    stream = torch.cuda.Stream()
    results2 = torch.zeros_like(result)

    # Performs the computing in a background stream
    def perform_computing():
        with torch.cuda.stream(stream):
            g2 = g.to(F.ctx())
        torch.cuda.current_stream().wait_stream(stream)
        g2.record_stream(torch.cuda.current_stream())
        torch.cuda._sleep(int(50 * cycles_per_ms))  # delay the computing
        results2.copy_(OPS.copy_u_sum(g2, x))

    perform_computing()
    with torch.cuda.stream(stream):
        # since we have called record stream for g2, g3 won't reuse its memory
        g3 = rand_graph(10, 20, device=F.ctx())
        g3.create_formats_()
    torch.cuda.current_stream().synchronize()
    assert torch.equal(result, results2)


@unittest.skipIf(
    F._default_context_str == "cpu", reason="stream only runs on GPU."
)
def test_record_stream_graph_negative():
    cycles_per_ms = _get_cycles_per_ms()

    g = rand_graph(10, 20, device=F.cpu())
    g.create_formats_()
    x = torch.ones(g.num_nodes(), 10).to(F.ctx())
    g1 = g.to(F.ctx())
    # this is necessary to initialize the cusparse handle
    result = OPS.copy_u_sum(g1, x)
    torch.cuda.current_stream().synchronize()

    stream = torch.cuda.Stream()
    results2 = torch.zeros_like(result)

    # Performs the computing in a background stream
    def perform_computing():
        with torch.cuda.stream(stream):
            g2 = g.to(F.ctx())
        torch.cuda.current_stream().wait_stream(stream)
        # omit record_stream will produce a wrong result
        # g2.record_stream(torch.cuda.current_stream())
        torch.cuda._sleep(int(50 * cycles_per_ms))  # delay the computing
        results2.copy_(OPS.copy_u_sum(g2, x))

    perform_computing()
    with torch.cuda.stream(stream):
        # g3 will reuse g2's memory block, resulting a wrong result
        g3 = rand_graph(10, 20, device=F.ctx())
        g3.create_formats_()
    torch.cuda.current_stream().synchronize()
    assert not torch.equal(result, results2)


if __name__ == "__main__":
    test_basics()
    test_set_get_stream()
    test_record_stream_ndarray()
    test_record_stream_graph_positive()
    test_record_stream_graph_negative()
