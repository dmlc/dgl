import torch

import backend as F

from dgl.mock_sparse import create_from_coo, diag, bspmm

def test_sparse_dense_mm():
    dev = F.ctx()
    # A: shape (N, M), X: shape (M, F)
    row = torch.tensor([0, 1, 1]).to(dev)
    col = torch.tensor([1, 0, 1]).to(dev)
    val = torch.randn(len(row)).to(dev)
    A = create_from_coo(row, col, val)
    X = torch.randn(2, 3).to(dev)
    sparse_result = A @ X
    dense_result = A.adj.to_dense() @ X
    assert torch.allclose(sparse_result, dense_result)

    # X: shape (M)
    X = torch.randn(2).to(dev)
    sparse_result = A @ X
    dense_result = A.adj.to_dense() @ X
    assert torch.allclose(sparse_result, dense_result)

def test_sparse_sparse_mm():
    dev = F.ctx()
    row1 = torch.tensor([0, 1, 1]).to(dev)
    col1 = torch.tensor([1, 0, 1]).to(dev)
    val1 = torch.randn(len(row1)).to(dev)
    A1 = create_from_coo(row1, col1, val1)

    row2 = torch.tensor([0, 1, 1]).to(dev)
    col2 = torch.tensor([0, 2, 1]).to(dev)
    val2 = torch.randn(len(row2)).to(dev)
    A2 = create_from_coo(row2, col2, val2)

    sparse_result = A1 @ A2
    dense_result = A1.adj.to_dense() @ A2.adj.to_dense()
    assert torch.allclose(sparse_result.adj.to_dense(), dense_result)

def test_sparse_diag_mm():
    dev = F.ctx()
    row = torch.tensor([0, 1, 1]).to(dev)
    col = torch.tensor([1, 0, 1]).to(dev)
    val1 = torch.randn(len(row)).to(dev)
    A = create_from_coo(row, col, val1)

    val2 = torch.randn(2).to(dev)
    D = diag(val2, (2, 3))
    M1 = (A @ D).adj
    M2 = (A @ D.as_sparse()).adj
    assert torch.allclose(M1.indices(), M2.indices())
    assert torch.allclose(M1.values(), M2.values())
    assert M1.shape == M2.shape

def test_diag_dense_mm():
    dev = F.ctx()
    # D: shape (N, N), X: shape (N, F)
    val = torch.randn(3)
    D = diag(val)
    X = torch.randn(3, 2)
    sparse_result = D @ X
    dense_result = D.as_sparse().adj.to_dense() @ X
    assert torch.allclose(sparse_result, dense_result)

    # D: shape (N, M), N > M, X: shape (M, F)
    val = torch.randn(3)
    D = diag(val, shape=(4, 3))
    sparse_result = D @ X
    dense_result = D.as_sparse().adj.to_dense() @ X
    assert torch.allclose(sparse_result, dense_result)

    # D: shape (N, M), N < M, X: shape (M, F)
    val = torch.randn(2)
    D = diag(val, shape=(2, 3))
    sparse_result = D @ X
    dense_result = D.as_sparse().adj.to_dense() @ X
    assert torch.allclose(sparse_result, dense_result)

    # D: shape (N, M), X: shape (M)
    val = torch.randn(3)
    D = diag(val)
    X = torch.randn(3)
    sparse_result = D @ X
    dense_result = D.as_sparse().adj.to_dense() @ X
    assert torch.allclose(sparse_result, dense_result)

def test_diag_sparse_mm():
    dev = F.ctx()
    row = torch.tensor([0, 1, 1]).to(dev)
    col = torch.tensor([1, 0, 1]).to(dev)
    val1 = torch.randn(len(row)).to(dev)
    A = create_from_coo(row, col, val1)

    val2 = torch.randn(2).to(dev)
    D = diag(val2, (3, 2))
    M1 = (D @ A).adj
    M2 = (D.as_sparse() @ A).adj
    assert torch.allclose(M1.indices(), M2.indices())
    assert torch.allclose(M1.values(), M2.values())
    assert M1.shape == M2.shape

def test_diag_diag_mm():
    # D1, D2: shape (N, N)
    val1 = torch.randn(3)
    D1 = diag(val1)
    val2 = torch.randn(3)
    D2 = diag(val2)
    sparse_result = D1 @ D2
    assert torch.allclose(sparse_result.val, D1.val * D2.val)

    # D1: shape (N, M), D2: shape (M, P)
    N = 3
    M = 4
    P = 2

    val1 = torch.randn(N)
    D1 = diag(val1, (N, M))
    val2 = torch.randn(P)
    D2 = diag(val2, (M, P))
    M1 = (D1 @ D2).as_sparse().adj
    M2 = (D1.as_sparse() @ D2.as_sparse()).adj
    assert torch.allclose(M1.indices(), M2.indices())
    assert torch.allclose(M1.values(), M2.values())
    assert M1.shape == M2.shape

def test_batch_sparse_dense_mm():
    dev = F.ctx()
    # A: shape (N, M), val shape (nnz, H)
    # X: shape (M, F)
    H = 4
    row = torch.tensor([0, 1, 1]).to(dev)
    col = torch.tensor([1, 0, 1]).to(dev)
    val = torch.randn(len(row), H).to(dev)
    A = create_from_coo(row, col, val)
    X = torch.randn(2, 3, H).to(dev)
    sparse_result = bspmm(A, X)
    dense_A = A.adj.to_dense()
    dense_result = torch.stack([
        dense_A[:, :, i] @ X[..., i] for i in range(H)
    ], dim=-1)
    assert torch.allclose(sparse_result, dense_result)
