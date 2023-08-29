import warnings

import backend as F
import pytest
import torch

from dgl.sparse import bspmm, diag, from_coo, val_like
from dgl.sparse.matmul import matmul

from .utils import (
    clone_detach_and_grad,
    dense_mask,
    rand_coo,
    rand_csc,
    rand_csr,
    rand_stride,
    sparse_matrix_to_dense,
    sparse_matrix_to_torch_sparse,
)


def _torch_sparse_mm(torch_A1, torch_A2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return torch.sparse.mm(torch_A1, torch_A2)


@pytest.mark.parametrize("create_func", [rand_coo, rand_csr, rand_csc])
@pytest.mark.parametrize("shape", [(2, 7), (5, 2)])
@pytest.mark.parametrize("nnz", [1, 10])
@pytest.mark.parametrize("out_dim", [None, 10])
def test_spmm(create_func, shape, nnz, out_dim):
    dev = F.ctx()
    A = create_func(shape, nnz, dev)
    if out_dim is not None:
        X = torch.randn(shape[1], out_dim, requires_grad=True, device=dev)
    else:
        X = torch.randn(shape[1], requires_grad=True, device=dev)

    X = rand_stride(X)
    sparse_result = matmul(A, X)
    grad = torch.randn_like(sparse_result)
    sparse_result.backward(grad)

    adj = sparse_matrix_to_dense(A)
    XX = clone_detach_and_grad(X)
    dense_result = torch.matmul(adj, XX)
    if out_dim is None:
        dense_result = dense_result.view(-1)
    dense_result.backward(grad)
    assert torch.allclose(sparse_result, dense_result, atol=1e-05)
    assert torch.allclose(X.grad, XX.grad, atol=1e-05)
    assert torch.allclose(
        dense_mask(adj.grad, A),
        sparse_matrix_to_dense(val_like(A, A.val.grad)),
        atol=1e-05,
    )


@pytest.mark.parametrize("create_func", [rand_coo, rand_csr, rand_csc])
@pytest.mark.parametrize("shape", [(2, 7), (5, 2)])
@pytest.mark.parametrize("nnz", [1, 10])
def test_bspmm(create_func, shape, nnz):
    dev = F.ctx()
    A = create_func(shape, nnz, dev, 2)
    X = torch.randn(shape[1], 10, 2, requires_grad=True, device=dev)
    X = rand_stride(X)

    sparse_result = matmul(A, X)
    grad = torch.randn_like(sparse_result)
    sparse_result.backward(grad)

    XX = clone_detach_and_grad(X)
    torch_A = A.to_dense().clone().detach().requires_grad_()
    torch_result = torch_A.permute(2, 0, 1) @ XX.permute(2, 0, 1)

    torch_result.backward(grad.permute(2, 0, 1))
    assert torch.allclose(
        sparse_result.permute(2, 0, 1), torch_result, atol=1e-05
    )
    assert torch.allclose(X.grad, XX.grad, atol=1e-05)
    assert torch.allclose(
        dense_mask(torch_A.grad, A),
        sparse_matrix_to_dense(val_like(A, A.val.grad)),
        atol=1e-05,
    )


@pytest.mark.parametrize("create_func1", [rand_coo, rand_csr, rand_csc])
@pytest.mark.parametrize("create_func2", [rand_coo, rand_csr, rand_csc])
@pytest.mark.parametrize("shape_n_m", [(5, 5), (5, 6)])
@pytest.mark.parametrize("shape_k", [3, 4])
@pytest.mark.parametrize("nnz1", [1, 10])
@pytest.mark.parametrize("nnz2", [1, 10])
def test_spspmm(create_func1, create_func2, shape_n_m, shape_k, nnz1, nnz2):
    dev = F.ctx()
    shape1 = shape_n_m
    shape2 = (shape_n_m[1], shape_k)
    A1 = create_func1(shape1, nnz1, dev)
    A2 = create_func2(shape2, nnz2, dev)
    A3 = matmul(A1, A2)
    grad = torch.randn_like(A3.val)
    A3.val.backward(grad)

    torch_A1 = sparse_matrix_to_torch_sparse(A1)
    torch_A2 = sparse_matrix_to_torch_sparse(A2)
    torch_A3 = _torch_sparse_mm(torch_A1, torch_A2)
    torch_A3_grad = sparse_matrix_to_torch_sparse(A3, grad)
    torch_A3.backward(torch_A3_grad)

    with torch.no_grad():
        assert torch.allclose(A3.to_dense(), torch_A3.to_dense(), atol=1e-05)
        assert torch.allclose(
            val_like(A1, A1.val.grad).to_dense(),
            torch_A1.grad.to_dense(),
            atol=1e-05,
        )
        assert torch.allclose(
            val_like(A2, A2.val.grad).to_dense(),
            torch_A2.grad.to_dense(),
            atol=1e-05,
        )


def test_spspmm_duplicate():
    dev = F.ctx()

    row = torch.tensor([1, 0, 0, 0, 1]).to(dev)
    col = torch.tensor([1, 1, 1, 2, 2]).to(dev)
    val = torch.randn(len(row)).to(dev)
    shape = (4, 4)
    A1 = from_coo(row, col, val, shape)

    row = torch.tensor([1, 0, 0, 1]).to(dev)
    col = torch.tensor([1, 1, 2, 2]).to(dev)
    val = torch.randn(len(row)).to(dev)
    shape = (4, 4)
    A2 = from_coo(row, col, val, shape)

    try:
        matmul(A1, A2)
    except:
        pass
    else:
        assert False, "Should raise error."

    try:
        matmul(A2, A1)
    except:
        pass
    else:
        assert False, "Should raise error."


@pytest.mark.parametrize("create_func", [rand_coo, rand_csr, rand_csc])
@pytest.mark.parametrize("sparse_shape", [(5, 5), (5, 6)])
@pytest.mark.parametrize("nnz", [1, 10])
def test_sparse_diag_mm(create_func, sparse_shape, nnz):
    dev = F.ctx()
    diag_shape = sparse_shape[1], sparse_shape[1]
    A = create_func(sparse_shape, nnz, dev)
    diag_val = torch.randn(sparse_shape[1], device=dev, requires_grad=True)
    D = diag(diag_val, diag_shape)
    B = matmul(A, D)
    grad = torch.randn_like(B.val)
    B.val.backward(grad)

    torch_A = sparse_matrix_to_torch_sparse(A)
    torch_D = sparse_matrix_to_torch_sparse(D)
    torch_B = _torch_sparse_mm(torch_A, torch_D)
    torch_B_grad = sparse_matrix_to_torch_sparse(B, grad)
    torch_B.backward(torch_B_grad)

    with torch.no_grad():
        assert torch.allclose(B.to_dense(), torch_B.to_dense(), atol=1e-05)
        assert torch.allclose(
            val_like(A, A.val.grad).to_dense(),
            torch_A.grad.to_dense(),
            atol=1e-05,
        )
        assert torch.allclose(
            diag(D.val.grad, D.shape).to_dense(),
            torch_D.grad.to_dense(),
            atol=1e-05,
        )


@pytest.mark.parametrize("create_func", [rand_coo, rand_csr, rand_csc])
@pytest.mark.parametrize("sparse_shape", [(5, 5), (5, 6)])
@pytest.mark.parametrize("nnz", [1, 10])
def test_diag_sparse_mm(create_func, sparse_shape, nnz):
    dev = F.ctx()
    diag_shape = sparse_shape[0], sparse_shape[0]
    A = create_func(sparse_shape, nnz, dev)
    diag_val = torch.randn(sparse_shape[0], device=dev, requires_grad=True)
    D = diag(diag_val, diag_shape)
    B = matmul(D, A)
    grad = torch.randn_like(B.val)
    B.val.backward(grad)

    torch_A = sparse_matrix_to_torch_sparse(A)
    torch_D = sparse_matrix_to_torch_sparse(D)
    torch_B = _torch_sparse_mm(torch_D, torch_A)
    torch_B_grad = sparse_matrix_to_torch_sparse(B, grad)
    torch_B.backward(torch_B_grad)

    with torch.no_grad():
        assert torch.allclose(B.to_dense(), torch_B.to_dense(), atol=1e-05)
        assert torch.allclose(
            val_like(A, A.val.grad).to_dense(),
            torch_A.grad.to_dense(),
            atol=1e-05,
        )
        assert torch.allclose(
            diag(D.val.grad, D.shape).to_dense(),
            torch_D.grad.to_dense(),
            atol=1e-05,
        )
