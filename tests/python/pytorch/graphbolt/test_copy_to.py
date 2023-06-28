import dgl.graphbolt
import unittest
import torch
import backend as F

@unittest.skipIf(
    F._default_context_str == "cpu", "CopyTo needs GPU to test"
)
def test_CopyTo():
    dp = dgl.graphbolt.MinibatchSampler(torch.randn(20), 4)
    dp = dgl.graphbolt.CopyTo(dp, "cuda")

    for data in dp:
        assert data.device.type == "cuda"
