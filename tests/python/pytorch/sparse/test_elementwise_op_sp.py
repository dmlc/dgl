import sys

import backend as F
import pytest
import torch

from dgl.sparse import div, from_coo, mul, power, spmatrix, val_like

from .utils import (
    rand_coo,
    rand_csc,
    rand_csr,
    rand_diag,
    sparse_matrix_to_dense,
)


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


@pytest.mark.parametrize(
    "create_func1", [rand_coo, rand_csr, rand_csc, rand_diag]
)
@pytest.mark.parametrize(
    "create_func2", [rand_coo, rand_csr, rand_csc, rand_diag]
)
@pytest.mark.parametrize("shape", [(5, 5), (5, 3)])
@pytest.mark.parametrize("nnz1", [5, 15])
@pytest.mark.parametrize("nnz2", [1, 14])
@pytest.mark.parametrize("nz_dim", [None, 3])
def test_spspmul(create_func1, create_func2, shape, nnz1, nnz2, nz_dim):
    dev = F.ctx()
    A = create_func1(shape, nnz1, dev, nz_dim)
    B = create_func2(shape, nnz2, dev, nz_dim)
    C = mul(A, B)
    assert not C.has_duplicate()

    DA = sparse_matrix_to_dense(A)
    DB = sparse_matrix_to_dense(B)
    DC = DA * DB

    grad = torch.rand_like(C.val)
    C.val.backward(grad)
    DC_grad = sparse_matrix_to_dense(val_like(C, grad))
    DC.backward(DC_grad)

    assert torch.allclose(sparse_matrix_to_dense(C), DC, atol=1e-05)
    assert torch.allclose(
        val_like(A, A.val.grad).to_dense(), DA.grad, atol=1e-05
    )
    assert torch.allclose(
        val_like(B, B.val.grad).to_dense(), DB.grad, atol=1e-05
    )


@pytest.mark.parametrize(
    "create_func", [rand_coo, rand_csr, rand_csc, rand_diag]
)
@pytest.mark.parametrize("shape", [(5, 5), (5, 3)])
@pytest.mark.parametrize("nnz", [1, 14])
@pytest.mark.parametrize("nz_dim", [None, 3])
def test_spspdiv(create_func, nnz, shape, nz_dim):
    dev = F.ctx()
    A = create_func(shape, nnz, dev, nz_dim)

    perm = torch.randperm(A.nnz, device=dev)
    rperm = torch.argsort(perm)
    B = spmatrix(A.indices()[:, perm], A.val[perm], A.shape)
    C = div(A, B)
    assert not C.has_duplicate()
    assert torch.allclose(C.val, A.val / B.val[rperm], atol=1e-05)
    assert torch.allclose(C.indices(), A.indices(), atol=1e-05)

    # No need to test backward here, since it is handled by Pytorch
