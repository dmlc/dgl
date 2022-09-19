import pytest
import dgl
import torch
from dgl.mock_sparse import SparseMatrix

parametrize_idtype = pytest.mark.parametrize(
    "idtype", [torch.int32, torch.int64]
)
parametrize_dtype = pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64]
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
    M = 10
    N = 50
    K = 20
    matB = torch.rand(M, K)
    matC = torch.rand(K, N)

    # A.val shape (nnz)
    val_scalar = torch.arange(1, 6).float()
    A = SparseMatrix(row, col, val_scalar, shape=(M, N))
    dgl_result = dgl.mock_sparse.sddmm(A, matB, matC)
    th_result = torch.sparse.sampled_addmm(A.adj.to_sparse_csr(), matB, matC)
    all_close_sparse(dgl_result.adj, th_result.to_sparse_coo())

    # A.val shape (nnz, D)
    v = torch.arange(1, 11).float().reshape(A.nnz, -1)
    A = SparseMatrix(row, col, v)
    dgl_result = dgl.mock_sparse.sddmm(A, matB, matC)

    th_result_i = []
    for i in range(A.val.shape[1]):
        A_coo_i = torch.sparse_coo_tensor(
            A.adj.indices(), A.val[:, i].contiguous().float()
        )
        A_csr_i = A_coo_i.to_sparse_csr()
        th_result_i.append(
            torch.sparse.sampled_addmm(A_csr_i, matB, matC).values()
        )
    th_result = torch.stack(th_result_i, dim=-1)
    torch.allclose(dgl_result.val, th_result)


if __name__ == "__main__":
    test_sddmm()
