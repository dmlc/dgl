import ctypes
from dgl._ffi import stream
import unittest
import backend as F
from dgl.utils import to_dgl_context


@unittest.skipIf(F._default_context_str == 'cpu', reason="stream only runs on GPU.")
def test_basics():
    ctx = to_dgl_context(F.ctx())
    prev_s = stream.current_stream(ctx)

    # create stream and set it as current
    s0 = stream.create_stream()
    assert s0.ctx == ctx
    assert s0.stream.value is not None
    stream.set_stream(s0)
    assert s0.stream.value == stream.current_stream(ctx).stream.value

    # create external stream and set it as current
    s1 = stream.ExternalStream(ctx, ctypes.c_void_p())
    assert s1.ctx == ctx
    assert s1.stream.value is None
    stream.set_stream(s1)
    assert s1.stream.value == stream.current_stream(ctx).stream.value

    # move data between CPU and GPU on specified stream
    s2 = stream.create_stream()
    stream.set_stream(s2)
    data_cpu = F.copy_to(F.randn((5, 6)), F.cpu())
    assert data_cpu.device == F.cpu()
    data_gpu = F.copy_to(data_cpu, F.ctx())
    data_cpu2 = F.copy_to(data_gpu, F.cpu())
    stream.synchronize_stream(s2)
    assert F.array_equal(data_cpu, data_cpu2)

    # restore original stream
    stream.set_stream(prev_s)


if __name__ == '__main__':
    test_basics()
