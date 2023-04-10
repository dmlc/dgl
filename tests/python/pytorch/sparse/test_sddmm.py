import sys

import backend as F
import pytest
import torch

from dgl.sparse import bsddmm, sddmm

from .utils import (
    clone_detach_and_grad,
    rand_coo,
    rand_csc,
    rand_csr,
    rand_stride,
)


@pytest.mark.parametrize("create_func", [rand_coo, rand_csr, rand_csc])
@pytest.mark.parametrize("shape", [(5, 5), (5, 4)])
@pytest.mark.parametrize("nnz", [2, 10])
@pytest.mark.parametrize("hidden", [1, 5])
def test_sddmm(create_func, shape, nnz, hidden):
    dev = F.ctx()
    A = create_func(shape, nnz, dev)
    if hidden > 1:
        B = torch.rand(shape[0], hidden, requires_grad=True, device=dev)
        C = torch.rand(hidden, shape[1], requires_grad=True, device=dev)
    else:
        B = torch.rand(shape[0], requires_grad=True, device=dev)
        C = torch.rand(shape[1], requires_grad=True, device=dev)

    B = rand_stride(B)
    C = rand_stride(C)

    A_val_clone = clone_detach_and_grad(A.val)
    dense_B = clone_detach_and_grad(B)
    dense_C = clone_detach_and_grad(C)

    sparse_result = sddmm(A, B, C)

    grad = torch.rand_like(sparse_result.val)
    sparse_result.val.backward(grad)

    if hidden == 1:
        dense_result = dense_B.view(-1, 1) @ dense_C.view(1, -1)
    else:
        dense_result = dense_B @ dense_C

    row, col = A.coo()
    dense_val = dense_result[row, col] * A_val_clone
    dense_val.backward(grad)

    assert torch.allclose(dense_val, sparse_result.val, atol=1e-05)
    assert torch.allclose(dense_C.grad, C.grad, atol=1e-05)
    assert torch.allclose(dense_B.grad, B.grad, atol=1e-05)
    assert torch.allclose(A_val_clone.grad, A.val.grad, atol=1e-05)


@pytest.mark.parametrize("create_func", [rand_coo, rand_csr, rand_csc])
@pytest.mark.parametrize("shape", [(5, 5), (5, 4)])
@pytest.mark.parametrize("nnz", [2, 10])
@pytest.mark.parametrize("nz_dim", [2, 10])
def test_bsddmm(create_func, shape, nnz, nz_dim):
    dev = F.ctx()
    hidden = 2
    A = create_func(shape, nnz, dev, nz_dim)
    B = torch.rand(shape[0], hidden, nz_dim, requires_grad=True, device=dev)
    C = torch.rand(hidden, shape[1], nz_dim, requires_grad=True, device=dev)

    B = rand_stride(B)
    C = rand_stride(C)

    A_val_clone = clone_detach_and_grad(A.val)
    dense_B = clone_detach_and_grad(B)
    dense_C = clone_detach_and_grad(C)

    sparse_result = bsddmm(A, B, C)

    grad = torch.rand_like(sparse_result.val)
    sparse_result.val.backward(grad)

    dense_result = dense_B.permute(2, 0, 1) @ dense_C.permute(2, 0, 1)
    dense_result = dense_result.permute(1, 2, 0)

    row, col = A.coo()
    dense_val = dense_result[row, col] * A_val_clone
    dense_val.backward(grad)

    assert torch.allclose(dense_val, sparse_result.val, atol=1e-05)
    assert torch.allclose(dense_C.grad, C.grad, atol=1e-05)
    assert torch.allclose(dense_B.grad, B.grad, atol=1e-05)
    assert torch.allclose(A_val_clone.grad, A.val.grad, atol=1e-05)
