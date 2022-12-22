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
