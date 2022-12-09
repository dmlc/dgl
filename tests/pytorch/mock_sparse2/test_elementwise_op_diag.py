import operator
import sys

import backend as F
import numpy as np
import pytest
import torch
from dgl.mock_sparse2 import diag

# TODO(#4818): Skipping tests on win.
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
    ctx = F.ctx()
    shape = (3, 4)
    D1 = diag(torch.arange(1, 4).to(ctx), shape=shape)
    D2 = diag(torch.arange(10, 13).to(ctx), shape=shape)
    result = op(D1, D2)
    assert torch.allclose(result.val, op(D1.val, D2.val), rtol=1e-4, atol=1e-4)
    assert result.shape == D1.shape


@pytest.mark.parametrize("v_scalar", [2, 2.5])
def test_diag_op_scalar(v_scalar):
    ctx = F.ctx()
    shape = (3, 4)
    D1 = diag(torch.arange(1, 4).to(ctx), shape=shape)

    # D * v
    D2 = D1 * v_scalar
    assert torch.allclose(D1.val * v_scalar, D2.val, rtol=1e-4, atol=1e-4)
    assert D1.shape == D2.shape

    # v * D
    D2 = v_scalar * D1
    assert torch.allclose(v_scalar * D1.val, D2.val, rtol=1e-4, atol=1e-4)
    assert D1.shape == D2.shape

    # D / v
    D2 = D1 / v_scalar
    assert torch.allclose(D1.val / v_scalar, D2.val, rtol=1e-4, atol=1e-4)
    assert D1.shape == D2.shape

    # D ^ v
    D1 = diag(torch.arange(1, 4).to(ctx))
    D2 = D1 ** v_scalar
    assert torch.allclose(D1.val ** v_scalar, D2.val, rtol=1e-4, atol=1e-4)
    assert D1.shape == D2.shape
