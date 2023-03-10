import unittest

import backend as F
import torch
import torch.distributed as dist

from dgl.cuda import nccl
from dgl.partition import NDArrayPartition


@unittest.skipIf(
    F._default_context_str == "cpu", reason="NCCL only runs on GPU."
)
def test_nccl_sparse_push_single_remainder():
    torch.cuda.set_device("cuda:0")
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12345",
        world_size=1,
        rank=0,
    )

    index = F.randint([10000], F.int32, F.ctx(), 0, 10000)
    value = F.uniform([10000, 100], F.float32, F.ctx(), -1.0, 1.0)

    part = NDArrayPartition(10000, 1, "remainder")

    ri, rv = nccl.sparse_all_to_all_push(index, value, part)
    assert F.array_equal(ri, index)
    assert F.array_equal(rv, value)

    dist.destroy_process_group()


@unittest.skipIf(
    F._default_context_str == "cpu", reason="NCCL only runs on GPU."
)
def test_nccl_sparse_pull_single_remainder():
    torch.cuda.set_device("cuda:0")
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12345",
        world_size=1,
        rank=0,
    )

    req_index = F.randint([10000], F.int64, F.ctx(), 0, 100000)
    value = F.uniform([100000, 100], F.float32, F.ctx(), -1.0, 1.0)

    part = NDArrayPartition(100000, 1, "remainder")

    rv = nccl.sparse_all_to_all_pull(req_index, value, part)
    exp_rv = F.gather_row(value, req_index)
    assert F.array_equal(rv, exp_rv)

    dist.destroy_process_group()


@unittest.skipIf(
    F._default_context_str == "cpu", reason="NCCL only runs on GPU."
)
def test_nccl_sparse_push_single_range():
    torch.cuda.set_device("cuda:0")
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12345",
        world_size=1,
        rank=0,
    )

    index = F.randint([10000], F.int32, F.ctx(), 0, 10000)
    value = F.uniform([10000, 100], F.float32, F.ctx(), -1.0, 1.0)

    part_ranges = F.copy_to(
        F.tensor([0, value.shape[0]], dtype=F.int64), F.ctx()
    )
    part = NDArrayPartition(10000, 1, "range", part_ranges=part_ranges)

    ri, rv = nccl.sparse_all_to_all_push(index, value, part)
    assert F.array_equal(ri, index)
    assert F.array_equal(rv, value)

    dist.destroy_process_group()


@unittest.skipIf(
    F._default_context_str == "cpu", reason="NCCL only runs on GPU."
)
def test_nccl_sparse_pull_single_range():
    torch.cuda.set_device("cuda:0")
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12345",
        world_size=1,
        rank=0,
    )

    req_index = F.randint([10000], F.int64, F.ctx(), 0, 100000)
    value = F.uniform([100000, 100], F.float32, F.ctx(), -1.0, 1.0)

    part_ranges = F.copy_to(
        F.tensor([0, value.shape[0]], dtype=F.int64), F.ctx()
    )
    part = NDArrayPartition(100000, 1, "range", part_ranges=part_ranges)

    rv = nccl.sparse_all_to_all_pull(req_index, value, part)
    exp_rv = F.gather_row(value, req_index)
    assert F.array_equal(rv, exp_rv)

    dist.destroy_process_group()


if __name__ == "__main__":
    test_nccl_sparse_push_single_remainder()
    test_nccl_sparse_pull_single_remainder()
    test_nccl_sparse_push_single_range()
    test_nccl_sparse_pull_single_range()
