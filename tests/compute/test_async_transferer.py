import dgl
import unittest
import backend as F

from dgl.dataloading import AsyncTransferer

def test_async_transferer_to_other():
    cpu_ones = F.ones([100,75,25], dtype=F.int32, ctx=F.cpu())
    tran = AsyncTransferer(F.ctx())
    t = tran.async_copy(cpu_ones, F.ctx())
    other_ones = t.wait()

    assert F.context(other_ones) == F.ctx()
    assert F.array_equal(F.copy_to(other_ones, ctx=F.cpu()), cpu_ones)

def test_async_transferer_from_other():
    other_ones = F.ones([100,75,25], dtype=F.int32, ctx=F.ctx())
    tran = AsyncTransferer(F.ctx())
    t = tran.async_copy(other_ones, F.cpu())
    cpu_ones = t.wait()

    assert F.context(cpu_ones) == F.cpu()
    assert F.array_equal(F.copy_to(other_ones, ctx=F.cpu()), cpu_ones)

if __name__ == '__main__':
    test_async_transferer_to_other()
    test_async_transferer_from_other()

