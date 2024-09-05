import unittest
import backend as F
import pytest
import dgl.graphbolt as gb
import torch

from functools import partial

WORLD_SIZE = 7

assert_equal = partial(torch.testing.assert_close, rtol=0, atol=0)

@unittest.skipIf(
    F._default_context_str != "gpu",
    reason="This test requires an NVIDIA GPU.",
)
@pytest.mark.parametrize("dtype",[torch.int32,torch.int64])
@pytest.mark.parametrize("rank", list(range(WORLD_SIZE)))
def test_gpu_cached_feature_read_async(dtype, rank):
    nodes_list1 = [torch.randint(0, 11111111, [777], dtype=dtype, device=F.ctx()) for i in range(10)]
    nodes_list2 = [nodes.sort()[0] for nodes in nodes_list1]

    res1 = torch.ops.graphbolt.rank_sort(nodes_list1, rank, WORLD_SIZE)
    res2 = torch.ops.graphbolt.rank_sort(nodes_list2, rank, WORLD_SIZE)

    for i, ((nodes1, idx1, offsets1), (nodes2, idx2, offsets2)) in enumerate(zip(res1, res2)):
        assert_equal(nodes_list1[i], nodes1[idx1.sort()[1]])
        assert_equal(nodes_list2[i], nodes2[idx2.sort()[1]])
        assert_equal(offsets1, offsets2)
        assert offsets1.is_pinned() and offsets2.is_pinned()
