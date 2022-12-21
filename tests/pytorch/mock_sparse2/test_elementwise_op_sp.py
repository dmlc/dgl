import operator
import sys

import backend as F
import pytest
import torch

from dgl.mock_sparse2 import create_from_coo, power

# TODO(#4818): Skipping tests on win.
if not sys.platform.startswith("linux"):
    pytest.skip("skipping tests on win", allow_module_level=True)


def all_close_sparse(A, row, col, val, shape):
    rowA, colA = A.coo()
    valA = A.val
    assert torch.allclose(rowA, row)
    assert torch.allclose(colA, col)
    assert torch.allclose(valA, val)
    assert A.shape == shape


@pytest.mark.parametrize("val_shape", [(3,), (3, 2)])
def test_pow(val_shape):
    # A ** v
    ctx = F.ctx()
    row = torch.tensor([1, 0, 2]).to(ctx)
    col = torch.tensor([0, 3, 2]).to(ctx)
    val = torch.randn(val_shape).to(ctx)
    A = create_from_coo(row, col, val, shape=(3, 4))
    exponent = 2
    A_new = A**exponent
    assert torch.allclose(A_new.val, val**exponent)
    assert A_new.shape == A.shape
    new_row, new_col = A_new.coo()
    assert torch.allclose(new_row, row)
    assert torch.allclose(new_col, col)

    # power(A, v)
    A_new = power(A, exponent)
    assert torch.allclose(A_new.val, val**exponent)
    assert A_new.shape == A.shape
    new_row, new_col = A_new.coo()
    assert torch.allclose(new_row, row)
    assert torch.allclose(new_col, col)
