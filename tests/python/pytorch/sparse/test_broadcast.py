import operator

import backend as F
import pytest
import torch

from dgl.sparse import sp_broadcast_v

from .utils import rand_coo


@pytest.mark.parametrize("shape", [(3, 4), (1, 5), (5, 1)])
@pytest.mark.parametrize("nnz", [1, 4])
@pytest.mark.parametrize("nz_dim", [None, 2])
@pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv"])
def test_sp_broadcast_v(shape, nnz, nz_dim, op):
    dev = F.ctx()
    A = rand_coo(shape, nnz, dev, nz_dim)

    v = torch.randn(A.shape[1], device=dev)
    res1 = sp_broadcast_v(A, v, op)
    if A.val.dim() == 1:
        rhs = v[A.col]
    else:
        rhs = v[A.col].view(-1, 1)
    res2 = getattr(operator, op)(A.val, rhs)
    assert torch.allclose(res1.val, res2)

    v = torch.randn(1, A.shape[1], device=dev)
    res1 = sp_broadcast_v(A, v, op)
    if A.val.dim() == 1:
        rhs = v.view(-1)[A.col]
    else:
        rhs = v.view(-1)[A.col].view(-1, 1)
    res2 = getattr(operator, op)(A.val, rhs)
    assert torch.allclose(res1.val, res2)

    v = torch.randn(A.shape[0], 1, device=dev)
    res1 = sp_broadcast_v(A, v, op)
    if A.val.dim() == 1:
        rhs = v.view(-1)[A.row]
    else:
        rhs = v.view(-1)[A.row].view(-1, 1)
    res2 = getattr(operator, op)(A.val, rhs)
    assert torch.allclose(res1.val, res2)
