import unittest

import backend as F
import dgl
import pytest
import torch
from dgl.mock_sparse import SparseMatrix

parametrize_idtype = pytest.mark.parametrize(
    "idtype", [torch.int32, torch.int64]
)
parametrize_dtype = pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64]
)


@unittest.skipIf(
    F.ctx().type == "cpu",
    reason="sddmm uses operator from pytorch which only supports CUDA",
)
def all_close_sparse(A, B):
    assert torch.allclose(A.indices(), B.indices())
    assert torch.allclose(A.values(), B.values())
    assert A.shape == B.shape


@parametrize_idtype
@parametrize_dtype
def test_sddmm(idtype, dtype):
    row = torch.tensor([1, 0, 2, 9, 1])
    col = torch.tensor([0, 49, 2, 1, 7])
    val = torch.arange(1, 6).float()
    A = SparseMatrix(row, col, val, shape=(10, 50))
    matB = torch.rand(10, 5)
    matC = torch.rand(5, 50)

    dgl_result = dgl.mock_sparse.sddmm(A, matB, matC)
    th_result = torch.sparse.sampled_addmm(A.adj.to_sparse_csr(), matB, matC)
    all_close_sparse(dgl_result.adj, th_result.to_sparse_coo())
