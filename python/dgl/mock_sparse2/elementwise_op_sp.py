"""DGL elementwise operators for sparse matrix module."""
from typing import Union
import torch

from .diag_matrix import DiagMatrix
from .sparse_matrix import SparseMatrix

__all__ = ["sp_add"]


def spsp_add(A, B):
    """ Invoke C++ sparse library for addition """
    return SparseMatrix(
        torch.ops.dgl_sparse.spsp_add(A.c_sparse_matrix, B.c_sparse_matrix)
    )


def sp_add(
    A: Union[SparseMatrix, DiagMatrix], B: Union[SparseMatrix, DiagMatrix]
) -> SparseMatrix:
    """Elementwise addition.

    Parameters
    ----------
    A : SparseMatrix or DiagMatrix
        Sparse matrix or diagonal matrix
    B : SparseMatrix or DiagMatrix
        Sparse matrix or diagonal matrix

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    Case 1: Add two sparse matrices of same sparsity structure

    >>> rowA = torch.tensor([1, 0, 2])
    >>> colA = torch.tensor([0, 3, 2])
    >>> valA = torch.tensor([10, 20, 30])
    >>> A = SparseMatrix(rowA, colA, valA, shape=(3, 4))
    >>> A + A
    SparseMatrix(indices=tensor([[0, 1, 2],
            [3, 0, 2]]),
    values=tensor([40, 20, 60]),
    shape=(3, 4), nnz=3)
    >>> w = torch.arange(1, len(rowA)+1)
    >>> A + A(w)
    SparseMatrix(indices=tensor([[0, 1, 2],
            [3, 0, 2]]),
    values=tensor([21, 12, 33]),
    shape=(3, 4), nnz=3)

    Case 2: Add two sparse matrices of different sparsity structure

    >>> rowB = torch.tensor([1, 2, 0, 2, 1])
    >>> colB = torch.tensor([0, 2, 1, 3, 3])
    >>> valB = torch.tensor([1, 2, 3, 4, 5])
    >>> B = SparseMatrix(rowB, colB, valB, shape=(3 ,4))
    >>> A + B
    SparseMatrix(indices=tensor([[0, 0, 1, 1, 2, 2],
            [1, 3, 0, 3, 2, 3]]),
    values=tensor([ 3, 20, 11,  5, 32,  4]),
    shape=(3, 4), nnz=6)

    Case 3: Add sparse matrix and diagonal matrix

    >>> D = diag(torch.arange(2, 5), shape=A.shape)
    >>> A + D
    SparseMatrix(indices=tensor([[0, 0, 1, 1, 2],
            [0, 3, 0, 1, 2]]),
    values=tensor([ 2, 20, 10,  3, 34]),
    shape=(3, 4), nnz=5)
    """
    B = B.as_sparse() if isinstance(B, DiagMatrix) else B
    if isinstance(A, SparseMatrix) and isinstance(B, SparseMatrix):
        return spsp_add(A, B)
    raise RuntimeError(
        "Elementwise addition between {} and {} is not "
        "supported.".format(type(A), type(B))
    )


SparseMatrix.__add__ = sp_add
SparseMatrix.__radd__ = sp_add
