import sys

import backend as F
import pytest
import torch

from dgl.sparse import from_coo

# TODO(#4818): Skipping tests on win.
if not sys.platform.startswith("linux"):
    pytest.skip("skipping tests on win", allow_module_level=True)


def test_neg():
    ctx = F.ctx()
    row = torch.tensor([1, 1, 3]).to(ctx)
    col = torch.tensor([1, 2, 3]).to(ctx)
    val = torch.tensor([1.0, 1.0, 2.0]).to(ctx)
    A = from_coo(row, col, val)
    neg_A = -A
    assert A.shape == neg_A.shape
    assert A.nnz == neg_A.nnz
    assert torch.allclose(-A.val, neg_A.val)
    assert torch.allclose(torch.stack(A.coo()), torch.stack(neg_A.coo()))
    assert A.val.device == neg_A.val.device
