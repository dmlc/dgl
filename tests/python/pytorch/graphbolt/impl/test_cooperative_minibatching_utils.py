import unittest

from functools import partial

import backend as F
import dgl.graphbolt as gb
import pytest
import torch

WORLD_SIZE = 7

assert_equal = partial(torch.testing.assert_close, rtol=0, atol=0)


@unittest.skipIf(
    F._default_context_str != "gpu",
    reason="This test requires an NVIDIA GPU.",
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rank", list(range(WORLD_SIZE)))
def test_rank_sort_and_unique_and_compact(dtype, rank):
    torch.manual_seed(7)
    nodes_list1 = [
        torch.randint(0, 2111111111, [777], dtype=dtype, device=F.ctx())
        for _ in range(10)
    ]
    nodes_list2 = [nodes.sort()[0] for nodes in nodes_list1]

    res1 = torch.ops.graphbolt.rank_sort(nodes_list1, rank, WORLD_SIZE)
    res2 = torch.ops.graphbolt.rank_sort(nodes_list2, rank, WORLD_SIZE)

    for i, ((nodes1, idx1, offsets1), (nodes2, idx2, offsets2)) in enumerate(
        zip(res1, res2)
    ):
        assert_equal(nodes_list1[i], nodes1[idx1])
        assert_equal(nodes_list2[i], nodes2[idx2])
        assert_equal(offsets1, offsets2)
        assert offsets1.is_pinned() and offsets2.is_pinned()

    res3 = torch.ops.graphbolt.rank_sort(nodes_list1, rank, WORLD_SIZE)

    # This function is deterministic. Call with identical arguments and check.
    for (nodes1, idx1, offsets1), (nodes3, idx3, offsets3) in zip(res1, res3):
        assert_equal(nodes1, nodes3)
        assert_equal(idx1, idx3)
        assert_equal(offsets1, offsets3)

    # The dependency on the rank argument is simply a permutation.
    res4 = torch.ops.graphbolt.rank_sort(nodes_list1, 0, WORLD_SIZE)
    for (nodes1, idx1, offsets1), (nodes4, idx4, offsets4) in zip(res1, res4):
        off1 = offsets1.tolist()
        off4 = offsets4.tolist()
        assert_equal(nodes1[idx1], nodes4[idx4])
        for i in range(WORLD_SIZE):
            j = (i - rank + WORLD_SIZE) % WORLD_SIZE
            assert_equal(
                nodes1[off1[j] : off1[j + 1]], nodes4[off4[i] : off4[i + 1]]
            )

    unique, compacted, offsets = gb.unique_and_compact(
        nodes_list1[:1], rank, WORLD_SIZE
    )

    nodes1, idx1, offsets1 = res1[0]

    assert_equal(unique, nodes1)
    assert_equal(compacted[0], idx1)
    assert_equal(offsets, offsets1)
