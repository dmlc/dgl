"""dgl sddmm operators for sparse matrix module."""
import torch
from .sp_matrix import SparseMatrix

__all__ = ['sddmm']

def sddmm(A: SparseMatrix, matB: torch.tensor, matC: torch.tensor) -> SparseMatrix:
    r""" Sampled-Dense-Dense Matrix Multiplication.

    .. math::
        out = (matB @ matC) * A

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix of shape (M, N)
    matB : Tensor
        Dense matrix of shape (M, K)
    matC : Tensor
        Dense matrix of shape (K, N)

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    Case 1: Sparse matrix with scalar values. `A.val.shape` is (nnz).

    >>> row = torch.tensor([1, 1, 2])
    >>> col = torch.tensor([2, 3, 3])
    >>> val = torch.arange(1, 4).float()
    >>> A = SparseMatrix(row, col, val, (3, 4))
    >>> mat1 = torch.randn(3, 5)
    >>> mat2 = torch.randn(5, 4)
    >>> dgl.mock_sparse.sddmm(A, mat1, mat2)
    SparseMatrix(indices=tensor([[1, 1, 2],
            [2, 3, 3]]),
    values=tensor([1.8035, 2.3375, 3.1255]),
    shape=(3, 4), nnz=3)

    Case 2: Sparse matrix with vectors values. `A.val.shape` is (nnz, D).

    >>> v = torch.arange(1, 7).float().reshape(A.nnz, -1)
    >>> A = SparseMatrix(row, col, v)
    >>> dgl.mock_sparse.sddmm(A, mat1, mat2)
    SparseMatrix(indices=tensor([[1, 1, 2],
            [2, 3, 3]]),
    values=tensor([[1.8035, 2.8035],
            [3.3375, 4.3375],
            [5.1255, 6.1255]]),
    shape=(3, 4), nnz=3)
    """
    # PyTorch's sddmm operator only supports CSR format
    if len(A.val.shape) == 1:
        # A.val is of shape (nnz)
        vals = torch.sparse.sampled_addmm(A.adj.to_sparse_csr(), matB, matC).values()
    else:
        # A.val is of shape (nnz, D)
        _, D = A.val.shape
        vals = []
        coo_index = A.adj.indices()
        A_csr = torch.sparse_coo_tensor(coo_index, A.val[:, 0]).to_sparse_csr()
        row_ptr = A_csr.crow_indices()
        col_indices = A_csr.col_indices()
        for i in range(D):
            val_i = A.val[:, i].contiguous().float()
            adj_i = torch.sparse_csr_tensor(row_ptr, col_indices, val_i)
            result_i = torch.sparse.sampled_addmm(adj_i, matB, matC)
            vals.append(result_i.values())
        vals = torch.stack(vals, dim=-1)
    return SparseMatrix(A.row, A.col, vals, A.adj.shape)
