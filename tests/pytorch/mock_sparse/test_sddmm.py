import numpy as np
import pytest
import dgl
import dgl.backend as F
import torch
import numpy
from dgl.mock_sparse import SparseMatrix
parametrize_idtype = pytest.mark.parametrize("idtype", [F.int32, F.int64])
parametrize_dtype = pytest.mark.parametrize('dtype', [F.float32, F.float64])

def all_close_sparse(A, B):
    assert torch.allclose(A.indices(), B.indices())
    assert torch.allclose(A.values(), B.values())
    assert A.shape == B.shape

@parametrize_idtype
@parametrize_dtype
def test_sddmm(idtype, dtype):
    M = 10
    N = 50
    K = 20
    rowA = torch.tensor([1, 0, 2, 7, 1])
    colA = torch.tensor([0, 49, 2, 1, 7])
    valA = torch.rand(len(rowA))
    A = SparseMatrix(rowA, colA, valA, shape=(M, N))
    matB = torch.rand(M, K)
    matC = torch.rand(K, N)

    out = dgl.mock_sparse.sddmm(A, matB, matC)
    torch_out = torch.sparse.sampled_addmm(A.adj.to_sparse_csr(), matB, matC)

    # all_close_sparse(out.adj, torch_out.to_sparse_coo())

if __name__ == '__main__':
    test_sddmm()
