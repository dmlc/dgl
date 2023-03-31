import sys

import backend as F
import torch

from dgl.sparse import diag, spmatrix


def test_neg():
    ctx = F.ctx()
    row = torch.tensor([1, 1, 3]).to(ctx)
    col = torch.tensor([1, 2, 3]).to(ctx)
    val = torch.tensor([1.0, 1.0, 2.0]).to(ctx)
    A = spmatrix(torch.stack([row, col]), val)
    neg_A = -A
    assert A.shape == neg_A.shape
    assert A.nnz == neg_A.nnz
    assert torch.allclose(-A.val, neg_A.val)
    assert torch.allclose(torch.stack(A.coo()), torch.stack(neg_A.coo()))
    assert A.val.device == neg_A.val.device


def test_diag_neg():
    ctx = F.ctx()
    val = torch.arange(3).float().to(ctx)
    D = diag(val)
    neg_D = -D
    assert D.shape == neg_D.shape
    assert torch.allclose(-D.val, neg_D.val)
    assert D.val.device == neg_D.val.device


def test_diag_inv():
    ctx = F.ctx()
    val = torch.arange(1, 4).float().to(ctx)
    D = diag(val)
    inv_D = D.inv()
    assert D.shape == inv_D.shape
    assert torch.allclose(1.0 / D.val, inv_D.val)
    assert D.val.device == inv_D.val.device
