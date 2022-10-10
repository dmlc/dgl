import unittest

import backend as F

from dgl.cuda import nccl
from dgl.partition import NDArrayPartition


def gen_test_id():
    return "{:0256x}".format(78236728318467363)


@unittest.skipIf(
    F._default_context_str == "cpu", reason="NCCL only runs on GPU."
)
def test_nccl_id():
    nccl_id = nccl.UniqueId()
    text = str(nccl_id)
    nccl_id2 = nccl.UniqueId(id_str=text)

    assert nccl_id == nccl_id2

    nccl_id2 = nccl.UniqueId(gen_test_id())

    assert nccl_id2 != nccl_id

    nccl_id3 = nccl.UniqueId(str(nccl_id2))

    assert nccl_id2 == nccl_id3


@unittest.skipIf(
    F._default_context_str == "cpu", reason="NCCL only runs on GPU."
)
def test_nccl_sparse_push_single_remainder():
    nccl_id = nccl.UniqueId()
    comm = nccl.Communicator(1, 0, nccl_id)

    index = F.randint([10000], F.int32, F.ctx(), 0, 10000)
    value = F.uniform([10000, 100], F.float32, F.ctx(), -1.0, 1.0)

    part = NDArrayPartition(10000, 1, "remainder")

    ri, rv = comm.sparse_all_to_all_push(index, value, part)
    assert F.array_equal(ri, index)
    assert F.array_equal(rv, value)


@unittest.skipIf(
    F._default_context_str == "cpu", reason="NCCL only runs on GPU."
)
def test_nccl_sparse_pull_single_remainder():
    nccl_id = nccl.UniqueId()
    comm = nccl.Communicator(1, 0, nccl_id)

    req_index = F.randint([10000], F.int64, F.ctx(), 0, 100000)
    value = F.uniform([100000, 100], F.float32, F.ctx(), -1.0, 1.0)

    part = NDArrayPartition(100000, 1, "remainder")

    rv = comm.sparse_all_to_all_pull(req_index, value, part)
    exp_rv = F.gather_row(value, req_index)
    assert F.array_equal(rv, exp_rv)


@unittest.skipIf(
    F._default_context_str == "cpu", reason="NCCL only runs on GPU."
)
def test_nccl_sparse_push_single_range():
    nccl_id = nccl.UniqueId()
    comm = nccl.Communicator(1, 0, nccl_id)

    index = F.randint([10000], F.int32, F.ctx(), 0, 10000)
    value = F.uniform([10000, 100], F.float32, F.ctx(), -1.0, 1.0)

    part_ranges = F.copy_to(
        F.tensor([0, value.shape[0]], dtype=F.int64), F.ctx()
    )
    part = NDArrayPartition(10000, 1, "range", part_ranges=part_ranges)

    ri, rv = comm.sparse_all_to_all_push(index, value, part)
    assert F.array_equal(ri, index)
    assert F.array_equal(rv, value)


@unittest.skipIf(
    F._default_context_str == "cpu", reason="NCCL only runs on GPU."
)
def test_nccl_sparse_pull_single_range():
    nccl_id = nccl.UniqueId()
    comm = nccl.Communicator(1, 0, nccl_id)

    req_index = F.randint([10000], F.int64, F.ctx(), 0, 100000)
    value = F.uniform([100000, 100], F.float32, F.ctx(), -1.0, 1.0)

    part_ranges = F.copy_to(
        F.tensor([0, value.shape[0]], dtype=F.int64), F.ctx()
    )
    part = NDArrayPartition(100000, 1, "range", part_ranges=part_ranges)

    rv = comm.sparse_all_to_all_pull(req_index, value, part)
    exp_rv = F.gather_row(value, req_index)
    assert F.array_equal(rv, exp_rv)


@unittest.skipIf(
    F._default_context_str == "cpu", reason="NCCL only runs on GPU."
)
def test_nccl_support():
    # this is just a smoke test, as we don't have any other way to know
    # if NCCL support is compiled in right now.
    nccl.is_supported()


if __name__ == "__main__":
    test_nccl_id()
    test_nccl_sparse_push_single()
    test_nccl_sparse_pull_single()
