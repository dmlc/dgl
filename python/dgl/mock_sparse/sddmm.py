"""Sampled Dense-Dense Matrix Multiplication (SDDMM) operator module."""
import torch

from .sp_matrix import SparseMatrix

__all__ = ["sddmm"]


def sddmm(
    A: SparseMatrix, mat1: torch.tensor, mat2: torch.tensor
) -> SparseMatrix:
    r"""Sampled-Dense-Dense Matrix Multiplication (sddmm). `sddmm` multiplies
    two dense matrices :attr:`mat1` and :attr:`mat2` at the nonzero locations
    of sparse matrix :attr:`A`.

    Mathematically `sddmm` is formulated as:

    .. math::
        out = (mat1 @ mat2) * A

    :attr:`A` can have scalar-shaped `(nnz)` or vector-shaped `(nnz, D)`
    nonzero values. For vector-shaped nonzero values, each `d` dimension in
    `D` performs a sddmm operation on :attr:`mat1` and :attr:`mat2`.


    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix of shape (M, N). Values can have `(nnz)` or `(nnz, D)`
        shape.
    mat1 : Tensor
        Dense matrix of shape `(M, K)`
    mat2 : Tensor
        Dense matrix of shape `(K, N)`

    Returns
    -------
    SparseMatrix
        Sparse matrix of shape `(M, N)`. Values can have `(nnz)` or `(nnz, D)`
        shape depending on the input :attr:`A`.

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
    # PyTorch's sddmm operator only supports CSR format.
    if len(A.val.shape) == 1:
        # A.val is of shape (nnz).
        vals = torch.sparse.sampled_addmm(
            A.adj.to_sparse_csr(), mat1, mat2
        ).values()
    else:
        # A.val is of shape (nnz, D).
        _, dim = A.val.shape
        vals = []
        coo_index = A.adj.indices()
        a_csr = torch.sparse_coo_tensor(coo_index, A.val[:, 0]).to_sparse_csr()
        row_ptr = a_csr.crow_indices()
        col_indices = a_csr.col_indices()
        for i in range(dim):
            val_i = A.val[:, i].contiguous().float()
            adj_i = torch.sparse_csr_tensor(row_ptr, col_indices, val_i)
            result_i = torch.sparse.sampled_addmm(adj_i, mat1, mat2)
            vals.append(result_i.values())
        vals = torch.stack(vals, dim=-1)
    return SparseMatrix(A.row, A.col, vals, A.adj.shape)
