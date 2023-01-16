import sys

import backend as F
import pytest
import torch

from dgl.sparse import from_coo, power

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


@pytest.mark.parametrize(
    "v_scalar", [2, 2.5, torch.tensor(2), torch.tensor(2.5)]
)
def test_muldiv_scalar(v_scalar):
    ctx = F.ctx()
    row = torch.tensor([1, 0, 2]).to(ctx)
    col = torch.tensor([0, 3, 2]).to(ctx)
    val = torch.randn(len(row)).to(ctx)
    A1 = from_coo(row, col, val, shape=(3, 4))

    # A * v
    A2 = A1 * v_scalar
    assert torch.allclose(A1.val * v_scalar, A2.val, rtol=1e-4, atol=1e-4)
    assert A1.shape == A2.shape

    # v * A
    A2 = v_scalar * A1
    assert torch.allclose(A1.val * v_scalar, A2.val, rtol=1e-4, atol=1e-4)
    assert A1.shape == A2.shape

    # A / v
    A2 = A1 / v_scalar
    assert torch.allclose(A1.val / v_scalar, A2.val, rtol=1e-4, atol=1e-4)
    assert A1.shape == A2.shape

    # v / A
    with pytest.raises(TypeError):
        v_scalar / A1


@pytest.mark.parametrize("val_shape", [(3,), (3, 2)])
def test_pow(val_shape):
    # A ** v
    ctx = F.ctx()
    row = torch.tensor([1, 0, 2]).to(ctx)
    col = torch.tensor([0, 3, 2]).to(ctx)
    val = torch.randn(val_shape).to(ctx)
    A = from_coo(row, col, val, shape=(3, 4))
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


@pytest.mark.parametrize("op", ["add", "sub"])
@pytest.mark.parametrize(
    "v_scalar", [2, 2.5, torch.tensor(2), torch.tensor(2.5)]
)
def test_error_op_scalar(op, v_scalar):
    ctx = F.ctx()
    row = torch.tensor([1, 0, 2]).to(ctx)
    col = torch.tensor([0, 3, 2]).to(ctx)
    val = torch.randn(len(row)).to(ctx)
    A = from_coo(row, col, val, shape=(3, 4))

    with pytest.raises(TypeError):
        A + v_scalar
    with pytest.raises(TypeError):
        v_scalar + A

    with pytest.raises(TypeError):
        A - v_scalar
    with pytest.raises(TypeError):
        v_scalar - A
