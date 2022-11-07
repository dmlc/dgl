import operator

import numpy as np
import pytest
import torch
import sys
from dgl.mock_sparse import diag

# FIXME: Skipping tests on win.
if not sys.platform.startswith("linux"):
    pytest.skip("skipping tests on win", allow_module_level=True)


def all_close_sparse(A, B):
    assert torch.allclose(A.indices(), B.indices())
    assert torch.allclose(A.values(), B.values())
    assert A.shape == B.shape


@pytest.mark.parametrize(
    "op", [operator.add, operator.sub, operator.mul, operator.truediv]
)
def test_diag_op_diag(op):
    D1 = diag(torch.arange(1, 4))
    D2 = diag(torch.arange(10, 13))
    assert np.allclose(op(D1, D2).val, op(D1.val, D2.val), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("v_scalar", [2, 2.5])
def test_diag_op_scalar(v_scalar):
    D1 = diag(torch.arange(1, 50))
    assert np.allclose(
        D1.val * v_scalar, (D1 * v_scalar).val, rtol=1e-4, atol=1e-4
    )
    assert np.allclose(
        v_scalar * D1.val, (D1 * v_scalar).val, rtol=1e-4, atol=1e-4
    )
    assert np.allclose(
        D1.val / v_scalar, (D1 / v_scalar).val, rtol=1e-4, atol=1e-4
    )
    assert np.allclose(
        pow(D1.val, v_scalar), pow(D1, v_scalar).val, rtol=1e-4, atol=1e-4
    )
