import sys

import backend as F
import pytest
import torch

from dgl.mock_sparse2 import create_from_coo, create_from_csc, create_from_csr

# TODO(#4818): Skipping tests on win.
if not sys.platform.startswith("linux"):
    pytest.skip("skipping tests on win", allow_module_level=True)


def get_adj(A):
    row, col = A.coo()
    edge_index = torch.cat((row.unsqueeze(0), col.unsqueeze(0)), 0)
    shape = A.shape
    val = A.val.detach()
    if len(A.val.shape) > 1:
        shape += (A.val.shape[-1],)
    return torch.sparse_coo_tensor(edge_index, val, shape).coalesce()


def test_spmm_coo():
    dev = F.ctx()
    # A: shape (N, M), X: shape (M, F)
    row = torch.tensor([0, 1, 1, 1]).to(dev)
    col = torch.tensor([1, 0, 1, 2]).to(dev)
    val = torch.randn(len(row), requires_grad=True, device=dev)
    A = create_from_coo(row, col, val)
    X = torch.randn(3, 4, requires_grad=True, device=dev)
    sparse_result = A @ X
    grad = torch.randn_like(sparse_result)
    sparse_result.backward(grad)

    adj = get_adj(A)
    adj.requires_grad_()
    XX = X.clone().detach()
    XX.requires_grad_()
    dense_result = torch.sparse.mm(adj, XX)
    dense_result.backward(grad)
    assert torch.allclose(sparse_result, dense_result)
    assert torch.allclose(X.grad, XX.grad)
    assert torch.allclose(adj.grad.coalesce().values(), val.grad)


def test_spmm_coo_one_dim_rhs():
    dev = F.ctx()
    # A: shape (N, M), X: shape (M,)
    row = torch.tensor([0, 1, 1, 1]).to(dev)
    col = torch.tensor([1, 0, 1, 2]).to(dev)
    val = torch.randn(len(row), requires_grad=True, device=dev)
    A = create_from_coo(row, col, val)
    X = torch.randn(3, requires_grad=True, device=dev)
    sparse_result = A @ X
    grad = torch.randn_like(sparse_result)
    sparse_result.backward(grad)

    adj = get_adj(A)
    adj.requires_grad_()
    XX = X.clone().detach()
    XX.requires_grad_()
    dense_result = torch.sparse.mm(adj, XX.view(-1, 1))
    dense_result = dense_result.view(-1)
    dense_result.backward(grad)
    assert torch.allclose(sparse_result, dense_result)
    assert torch.allclose(X.grad, XX.grad)
    assert torch.allclose(adj.grad.coalesce().values(), val.grad)


def test_spmm_csr():
    dev = F.ctx()
    # A: shape (N, M), X: shape (M, F)
    indptr = torch.tensor([0, 1, 4]).to(dev)
    indices = torch.tensor([1, 0, 1, 2]).to(dev)
    val = torch.randn(len(indices), requires_grad=True, device=dev)
    A = create_from_csr(indptr, indices, val, shape=(2, 3))
    X = torch.randn(3, 4, requires_grad=True, device=dev)
    sparse_result = A @ X
    grad = torch.randn_like(sparse_result)
    sparse_result.backward(grad)

    adj = get_adj(A)
    adj.requires_grad_()
    XX = X.clone().detach()
    XX.requires_grad_()
    dense_result = torch.sparse.mm(adj, XX)
    dense_result.backward(grad)
    assert torch.allclose(sparse_result, dense_result)
    assert torch.allclose(X.grad, XX.grad)
    assert torch.allclose(adj.grad.coalesce().values(), val.grad)


def test_spmm_csc():
    dev = F.ctx()
    # A: shape (N, M), X: shape (M, F)
    indptr = torch.tensor([0, 1, 3, 4]).to(dev)
    indices = torch.tensor([0, 0, 1, 1]).to(dev)
    val = torch.randn(len(indices), requires_grad=True, device=dev)
    A = create_from_csc(indptr, indices, val, shape=(2, 3))
    X = torch.randn(3, 4, requires_grad=True, device=dev)
    sparse_result = A @ X
    grad = torch.randn_like(sparse_result)
    sparse_result.backward(grad)

    adj = get_adj(A)
    adj.requires_grad_()
    XX = X.clone().detach()
    XX.requires_grad_()
    dense_result = torch.sparse.mm(adj, XX)
    dense_result.backward(grad)
    assert torch.allclose(sparse_result, dense_result)
    assert torch.allclose(X.grad, XX.grad)
    assert torch.allclose(adj.grad.coalesce().values(), val.grad)
