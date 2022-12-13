"""DGL elementwise operators for sparse matrix module."""
import torch

from .sparse_matrix import SparseMatrix

__all__ = ["sp_add"]


def spsp_add(A, B):
    """Invoke C++ sparse library for addition"""
    return SparseMatrix(
        torch.ops.dgl_sparse.spsp_add(A.c_sparse_matrix, B.c_sparse_matrix)
    )


def sp_add(A: SparseMatrix, B: SparseMatrix) -> SparseMatrix:
    """Elementwise addition.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    B : SparseMatrix
        Sparse matrix

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------

    >>> row = torch.tensor([1, 0, 2])
    >>> col = torch.tensor([0, 3, 2])
    >>> val = torch.tensor([10, 20, 30])
    >>> A = create_from_coo(row, col, val, shape=(3, 4))
    >>> A + A
    SparseMatrix(indices=tensor([[0, 1, 2],
            [3, 0, 2]]),
    values=tensor([40, 20, 60]),
    shape=(3, 4), nnz=3)
    """
    if isinstance(A, SparseMatrix) and isinstance(B, SparseMatrix):
        return spsp_add(A, B)
    raise RuntimeError(
        "Elementwise addition between {} and {} is not "
        "supported.".format(type(A), type(B))
    )


SparseMatrix.__add__ = sp_add
SparseMatrix.__radd__ = sp_add
