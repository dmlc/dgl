import operator

import numpy as np
import pytest
import sys
import torch

import dgl

if sys.platform.startswith("linux"):
    from dgl.mock_sparse2 import create_from_coo


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Sparse library only supports linux for now",
)
def test_sparse_op_sparse():
    rowA = torch.tensor([1, 0, 2, 7, 1])
    colA = torch.tensor([0, 49, 2, 1, 7])
    valA = torch.rand(len(rowA))
    A = create_from_coo(rowA, colA, valA, (10, 50))
    valB = torch.rand(len(rowA))
    B = create_from_coo(rowA, colA, valB, (10, 50))

    C = A + B
    torch.allclose(C.val, valA + valB)
