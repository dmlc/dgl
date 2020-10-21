import dgl
import unittest
import backend as F

from dgl.sampling import AsyncTransferer

@unittest.skipIf(F._default_context_str == 'cpu', reason="Async transfer tests require a GPU to transfer to.")
def async_transferer_to_gpu():
    cpu_ones = F.ones([100,75,25], dtype=F.int32).cpu()
    tran = AsyncTransferer(F.ctx())
    t = tran.async_copy(cpu_ones, F.ctx())
    gpu_ones = t.wait()

    assert F.ctx(gpu_ones) == F.ctx()
    assert F.equal(gpu_ones.cpu(), cpu_ones)

@unittest.skipIf(F._default_context_str == 'cpu', reason="Async transfer tests require a GPU to transfer from.")
def async_transferer_to_gpu():
    gpu_ones = F.ones([100,75,25], dtype=F.int32, ctx=F.ctx())
    tran = AsyncTransferer(F.ctx())
    t = tran.async_copy(gpu_ones, F.cpu())
    cpu_ones = t.wait()

    assert F.ctx(cpu_ones) == F.cpu()
    assert F.equal(gpu_ones.cpu(), cpu_ones)

if __name__ == '__main__':
    test_async_transferer()

