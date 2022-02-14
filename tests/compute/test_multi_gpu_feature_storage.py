##
#   Copyright 2021 Contributors 
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import unittest
import backend as F
import dgl
from dgl.contrib.multi_gpu_feature_storage import MultiGPUFeatureStorage
from dgl.partition import NDArrayPartition


class DummyCommunicator:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def sparse_all_to_all_pull(self, req_idx, value, partition):
        # assume all indices are local
        idxs = partition.map_to_local(req_idx)
        return F.gather_row(value, idxs)

    def rank(self):
        return self._rank

    def size(self):
        return self._size

@unittest.skipIf(F._default_context_str == 'cpu',
                 reason="MutliGPUTensor only works on GPUs.")
def test_get_global_1part():
    t = F.copy_to(F.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]), F.ctx())
    comm = DummyCommunicator(0, 1)
    partition = NDArrayPartition(len(t), comm.size(), mode='remainder')
    mt = MultiGPUFeatureStorage(t.shape, t.dtype, F.ctx(), comm, partition)
    mt.all_set_global(t)

    idxs = F.copy_to(F.tensor([1,3], dtype=F.int64), ctx=F.ctx())
    act = mt.all_gather_row(idxs)
    exp = F.gather_row(t, idxs)

    assert F.array_equal(exp, act)

@unittest.skipIf(F._default_context_str == 'cpu',
                 reason="MutliGPUTensor only works on GPUs.")
def test_get_global_3part():
    t = F.copy_to(F.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]), F.ctx())
    comm = DummyCommunicator(2, 3)
    partition = NDArrayPartition(len(t), comm.size(), mode='remainder')
    mt = MultiGPUFeatureStorage(t.shape, t.dtype, F.ctx(), comm, partition)
    mt.all_set_global(t)

    idxs = F.copy_to(F.tensor([2], dtype=F.int64), ctx=F.ctx())
    act = mt.all_gather_row(idxs)
    exp = F.gather_row(t, idxs)

    assert F.array_equal(exp, act)

@unittest.skipIf(F._default_context_str == 'cpu',
                 reason="MutliGPUTensor only works on GPUs.")
def test_fetch_3part():
    t = F.copy_to(F.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]), F.ctx())
    comm = DummyCommunicator(2, 3)
    partition = NDArrayPartition(len(t), comm.size(), mode='remainder')
    mt = MultiGPUFeatureStorage(t.shape, t.dtype, F.ctx(), comm, partition)
    mt.all_set_global(t)

    idxs = F.copy_to(F.tensor([2], dtype=F.int64), ctx=F.ctx())
    act = mt.fetch(idxs, F.ctx())
    exp = F.gather_row(t, idxs)

    assert F.array_equal(exp, act)

@unittest.skipIf(F._default_context_str == 'cpu',
                 reason="MutliGPUTensor only works on GPUs.")
def test_requires_ddp():
    t = F.copy_to(F.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]), F.ctx())
    comm = DummyCommunicator(2, 3)
    partition = NDArrayPartition(len(t), comm.size(), mode='remainder')
    mt = MultiGPUFeatureStorage(t.shape, t.dtype, F.ctx(), comm, partition)
    assert mt.requires_ddp()

    comm = DummyCommunicator(0, 1)
    partition = NDArrayPartition(len(t), comm.size(), mode='remainder')
    mt = MultiGPUFeatureStorage(t.shape, t.dtype, F.ctx(), comm, partition)
    assert not mt.requires_ddp()

@unittest.skipIf(F._default_context_str == 'cpu',
                 reason="MutliGPUTensor only works on GPUs.")
def test_get_local_1part():
    t = F.copy_to(F.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]), F.ctx())
    comm = DummyCommunicator(0, 1)
    partition = NDArrayPartition(len(t), comm.size(), mode='remainder')
    mt = MultiGPUFeatureStorage(t.shape, t.dtype, F.ctx(), comm, partition)
    mt.all_set_global(t)

    act = mt.get_local()
    exp = t

    assert F.array_equal(exp, act)

@unittest.skipIf(F._default_context_str == 'cpu',
                 reason="MutliGPUTensor only works on GPUs.")
def test_get_local_3part():
    t = F.copy_to(F.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]), F.ctx())
    comm = DummyCommunicator(2, 3)
    partition = NDArrayPartition(len(t), comm.size(), mode='remainder')
    mt = MultiGPUFeatureStorage(t.shape, t.dtype, F.ctx(), comm, partition)
    mt.all_set_global(t)

    act = mt.get_local()
    exp = F.gather_row(t, F.tensor([2], dtype=F.int64))

    assert F.array_equal(exp, act)

@unittest.skipIf(F._default_context_str == 'cpu',
                 reason="MutliGPUTensor only works on GPUs.")
def test_set_local_1part():
    t = F.copy_to(F.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]), F.ctx())
    comm = DummyCommunicator(0, 1)
    partition = NDArrayPartition(len(t), comm.size(), mode='remainder')
    mt = MultiGPUFeatureStorage(t.shape, t.dtype, F.ctx(), comm, partition)
    t_local = t
    mt.set_local(t_local)

    act = mt.get_local()
    exp = t_local

    assert F.array_equal(exp, act)

@unittest.skipIf(F._default_context_str == 'cpu',
                 reason="MutliGPUTensor only works on GPUs.")
def test_set_local_3part():
    t = F.copy_to(F.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]), F.ctx())
    comm = DummyCommunicator(2, 3)
    partition = NDArrayPartition(len(t), comm.size(), mode='remainder')
    mt = MultiGPUFeatureStorage(t.shape, t.dtype, F.ctx(), comm, partition)
    t_local = F.gather_row(t, F.tensor([2], dtype=F.int64))
    mt.set_local(t_local)

    act = mt.get_local()
    exp = t_local

    assert F.array_equal(exp, act)

@unittest.skipIf(F._default_context_str == 'cpu',
                 reason="MutliGPUTensor only works on GPUs.")
def test_backend():
    t = F.copy_to(F.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]), F.ctx())
    comm = DummyCommunicator(2, 3)
    partition = NDArrayPartition(len(t), comm.size(), mode='remainder')
    mt = MultiGPUFeatureStorage(t.shape, t.dtype, F.ctx(), comm, partition)
    t_local = F.gather_row(t, F.tensor([2], dtype=F.int64))
    mt.set_local(t_local)

    assert F.ctx() == F.context(mt)
    assert F.shape(t) == F.shape(mt)
    assert F.dtype(t) == F.dtype(mt)


if __name__ == '__main__':
    test_get_global_1part()
    test_get_global_3part()
    test_get_local_1part()
    test_get_local_3part()
    test_set_local_1part()
    test_set_local_3part()
    test_backend()
