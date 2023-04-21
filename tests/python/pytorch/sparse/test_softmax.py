import sys

import backend as F

import dgl
import pytest
import torch
from dgl.sparse import from_coo, softmax


@pytest.mark.parametrize("val_D", [None, 2])
@pytest.mark.parametrize("csr", [True, False])
@pytest.mark.parametrize("dim", [0, 1])
def test_softmax(val_D, csr, dim):
    dev = F.ctx()
    row = torch.tensor([0, 0, 1, 1]).to(dev)
    col = torch.tensor([0, 2, 1, 2]).to(dev)
    nnz = len(row)
    if val_D is None:
        val = torch.randn(nnz).to(dev)
    else:
        val = torch.randn(nnz, val_D).to(dev)

    val_sparse = val.clone().requires_grad_()
    A = from_coo(row, col, val_sparse)

    if csr:
        # Test CSR
        A.csr()

    A_max = softmax(A, dim)
    if dim == 1:
        g = dgl.graph((col, row), num_nodes=max(A.shape))
    else:
        g = dgl.graph((row, col), num_nodes=max(A.shape))
    val_g = val.clone().requires_grad_()
    score = dgl.nn.functional.edge_softmax(g, val_g)
    assert torch.allclose(A_max.val, score, atol=1e-05)

    grad = torch.randn_like(score).to(dev)
    A_max.val.backward(grad)
    score.backward(grad)
    assert torch.allclose(A.val.grad, val_g.grad, atol=1e-05)
