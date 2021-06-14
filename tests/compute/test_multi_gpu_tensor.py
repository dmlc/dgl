import unittest
import backend as F
import dgl
from dgl.distributed.multi_gpu_tensor import MultiGPUTensor


class DummyCommunicator:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size


    def sparse_all_to_all_pull(self, req_idx, value, partition):
        # assume all indices are local
        idxs = partition.map_to_local(req_idx)
        return value[idxs]

    def rank(self):
        return self._rank

    def size(self):
        return self._size

@unittest.skipIf(F._default_context_str == 'cpu',
                 reason="MutliGPUTensor only works on GPUs.")
def test_get_global_1part():
    t = F.copy_to(F.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]), F.ctx())
    comm = DummyCommunicator(0, 1)
    mt = MultiGPUTensor(t.shape, t.dtype, F.ctx(), comm)
    mt.set_local(t)

    idxs = F.copy_to(F.tensor([1,3], dtype=F.int64), ctx=F.ctx())
    act = mt.get_global(idxs)
    exp = t[idxs]

    assert F.array_equal(exp, act)


if __name__ == '__main__':
    test_get_global_1part()
