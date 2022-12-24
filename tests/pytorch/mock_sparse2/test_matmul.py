import sys

import backend as F
import pytest
import torch

from dgl.mock_sparse2 import val_like

from .utils import (
    clone_detach_and_grad,
    rand_coo,
    rand_csc,
    rand_csr,
    sparse_matrix_to_dense,
    sparse_matrix_to_torch_sparse,
)

# TODO(#4818): Skipping tests on win.
if not sys.platform.startswith("linux"):
    pytest.skip("skipping tests on win", allow_module_level=True)


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

    sparse_result = A @ X
    grad = torch.randn_like(sparse_result)
    sparse_result.backward(grad)

    adj = sparse_matrix_to_torch_sparse(A)
    XX = clone_detach_and_grad(X)
    torch_sparse_result = torch.sparse.mm(
        adj, XX.view(-1, 1) if out_dim is None else XX
    )
    if out_dim is None:
        torch_sparse_result = torch_sparse_result.view(-1)
    torch_sparse_result.backward(grad)
    assert torch.allclose(sparse_result, torch_sparse_result, atol=1e-05)
    assert torch.allclose(X.grad, XX.grad, atol=1e-05)
    assert torch.allclose(
        adj.grad.coalesce().to_dense(),
        sparse_matrix_to_dense(val_like(A, A.val.grad)),
        atol=1e-05,
    )


@pytest.mark.parametrize("create_func1", [rand_coo, rand_csr, rand_csc])
@pytest.mark.parametrize("create_func2", [rand_coo, rand_csr, rand_csc])
@pytest.mark.parametrize("shape_n_m", [(5, 5), (5, 6)])
@pytest.mark.parametrize("shape_k", [3, 4])
@pytest.mark.parametrize("nnz1", [1, 10])
@pytest.mark.parametrize("nnz2", [1, 10])
def test_sparse_sparse_mm(
    create_func1, create_func2, shape_n_m, shape_k, nnz1, nnz2
):
    dev = F.ctx()
    shape1 = shape_n_m
    shape2 = (shape_n_m[1], shape_k)
    A1 = create_func1(shape1, nnz1, dev)
    A2 = create_func2(shape2, nnz2, dev)
    A3 = A1 @ A2
    grad = torch.randn_like(A3.val)
    A3.val.backward(grad)

    torch_A1 = sparse_matrix_to_torch_sparse(A1)
    torch_A2 = sparse_matrix_to_torch_sparse(A2)
    torch_A3 = torch.sparse.mm(torch_A1, torch_A2)
    torch_A3_grad = sparse_matrix_to_torch_sparse(A3, grad)
    torch_A3.backward(torch_A3_grad)

    with torch.no_grad():
        assert torch.allclose(A3.dense(), torch_A3.to_dense(), atol=1e-05)
        assert torch.allclose(
            val_like(A1, A1.val.grad).dense(),
            torch_A1.grad.to_dense(),
            atol=1e-05,
        )
        assert torch.allclose(
            val_like(A2, A2.val.grad).dense(),
            torch_A2.grad.to_dense(),
            atol=1e-05,
        )
