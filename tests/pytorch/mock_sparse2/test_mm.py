import torch

import backend as F
import sys
import pytest

from dgl.mock_sparse2 import create_from_coo

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


def test_sparse_dense_mm():
    dev = F.ctx()
    # A: shape (N, M), X: shape (M, F)
    row = torch.tensor([0, 1, 1]).to(dev)
    col = torch.tensor([1, 0, 1]).to(dev)
    val = torch.randn(len(row)).to(dev)
    A = create_from_coo(row, col, val)
    X = torch.randn(2, 3).to(dev)
    sparse_result = A @ X

    adj = get_adj(A)
    dense_result = torch.sparse.mm(adj, X)
    assert torch.allclose(sparse_result, dense_result)

    # X: shape (M)
    X = torch.randn(2).to(dev)
    sparse_result = A @ X
    dense_result = adj @ X
    assert torch.allclose(sparse_result, dense_result)


def test_sparse_dense_mm_autograd():
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


def test_sparse_dense_mm_autograd2():
    dev = F.ctx()
    # A: shape (N, M), X: shape (M, F)
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
