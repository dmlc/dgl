import dgl
import unittest
import backend as F

from dgl.contrib.sampling import AsyncTransferer

@unittest.skipIf(F._default_context_str == 'cpu', reason="Async transfer tests require a GPU to transfer to.")
def async_transferer():
    cpu_ones = F.ones([100,75,25], dtype=F.int32).cpu()

    tran = AsyncTransferer(F.ctx())

    t = tran.async_copy(cpu_ones, F.ctx())

    gpu_ones = t.wait()

    assert F.ctx(gpu_ones) == F.ctx()


if __name__ == '__main__':
    test_async_transferer()

