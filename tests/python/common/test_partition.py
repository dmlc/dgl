import unittest

import backend as F

from dgl.distributed import graph_partition_book as gpb
from dgl.partition import NDArrayPartition
from utils import parametrize_idtype


@unittest.skipIf(
    F._default_context_str == "cpu",
    reason="NDArrayPartition only works on GPU.",
)
@parametrize_idtype
def test_get_node_partition_from_book(idtype):
    node_map = {"_N": F.tensor([[0, 3], [4, 5], [6, 10]], dtype=idtype)}
    edge_map = {
        ("_N", "_E", "_N"): F.tensor([[0, 9], [10, 15], [16, 25]], dtype=idtype)
    }
    ntypes = {ntype: i for i, ntype in enumerate(node_map)}
    etypes = {etype: i for i, etype in enumerate(edge_map)}
    book = gpb.RangePartitionBook(0, 3, node_map, edge_map, ntypes, etypes)
    partition = gpb.get_node_partition_from_book(book, F.ctx())
    assert partition.num_parts() == 3
    assert partition.array_size() == 11

    # Test map_to_local
    test_ids = F.copy_to(F.tensor([0, 2, 6, 7, 10], dtype=idtype), F.ctx())
    act_ids = partition.map_to_local(test_ids)
    exp_ids = F.copy_to(F.tensor([0, 2, 0, 1, 4], dtype=idtype), F.ctx())
    assert F.array_equal(act_ids, exp_ids)

    # Test map_to_global
    test_ids = F.copy_to(F.tensor([0, 2], dtype=idtype), F.ctx())
    act_ids = partition.map_to_global(test_ids, 0)
    exp_ids = F.copy_to(F.tensor([0, 2], dtype=idtype), F.ctx())
    assert F.array_equal(act_ids, exp_ids)

    test_ids = F.copy_to(F.tensor([0, 1], dtype=idtype), F.ctx())
    act_ids = partition.map_to_global(test_ids, 1)
    exp_ids = F.copy_to(F.tensor([4, 5], dtype=idtype), F.ctx())
    assert F.array_equal(act_ids, exp_ids)

    test_ids = F.copy_to(F.tensor([0, 1, 4], dtype=idtype), F.ctx())
    act_ids = partition.map_to_global(test_ids, 2)
    exp_ids = F.copy_to(F.tensor([6, 7, 10], dtype=idtype), F.ctx())
    assert F.array_equal(act_ids, exp_ids)

    # Test generate_permutation
    test_ids = F.copy_to(F.tensor([6, 0, 7, 2, 10], dtype=idtype), F.ctx())
    perm, split_sum = partition.generate_permutation(test_ids)
    exp_perm = F.copy_to(F.tensor([1, 3, 0, 2, 4], dtype=idtype), F.ctx())
    exp_sum = F.copy_to(F.tensor([2, 0, 3]), F.ctx())
    assert F.array_equal(perm, exp_perm)
    assert F.array_equal(split_sum, exp_sum)
