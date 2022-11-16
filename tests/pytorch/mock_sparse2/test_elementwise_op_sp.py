import operator

import numpy as np
import pytest
import torch
import sys
import dgl
from dgl.mock_sparse2 import create_from_coo, diag

# FIXME: Skipping tests on win.
if not sys.platform.startswith("linux"):
    pytest.skip("skipping tests on win", allow_module_level=True)

def all_close_sparse(A, row, col, val, shape):
    rowA, colA, valA = A.coo()
    assert torch.allclose(rowA, row)
    assert torch.allclose(colA, col)
    assert torch.allclose(valA, val)
    assert A.shape == shape


@pytest.mark.parametrize("op", [operator.add])
def test_sparse_op_sparse(op):
    rowA = torch.tensor([1, 0, 2, 7, 1])
    colA = torch.tensor([0, 49, 2, 1, 7])
    valA = torch.rand(len(rowA))
    A = create_from_coo(rowA, colA, valA, shape=(10, 50))
    w = torch.rand(len(rowA))
    A1 = create_from_coo(rowA, colA, w, shape=(10, 50))

    def _test():
        all_close_sparse(op(A, A1), rowA, colA, valA + w, (10, 50))

    _test()


@pytest.mark.skip(
    reason="No way to test it because we does not element-wise op \
    between matrices with different sparsity"
)
@pytest.mark.parametrize("op", [operator.add])
def test_sparse_op_diag(op):
    rowA = torch.tensor([1, 0, 2, 7, 1])
    colA = torch.tensor([0, 49, 2, 1, 7])
    valA = torch.rand(len(rowA))
    A = create_from_coo(rowA, colA, valA, shape=(10, 50))
    D = diag(torch.arange(2, 12), shape=A.shape)
    D_sp = D.as_sparse()

    def _test():
        all_close_sparse(op(A, D), *D_sp.coo(), [10, 50])

    _test()
