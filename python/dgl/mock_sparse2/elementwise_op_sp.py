"""DGL elementwise operators for sparse matrix module."""
from typing import Union

import torch

from .diag_matrix import DiagMatrix
from .sparse_matrix import SparseMatrix, val_like

__all__ = ["sp_add", "sp_power"]


def spsp_add(A, B):
    """Invoke C++ sparse library for addition"""
    return SparseMatrix(
        torch.ops.dgl_sparse.spsp_add(A.c_sparse_matrix, B.c_sparse_matrix)
    )


def sp_add(A: SparseMatrix, B: Union[DiagMatrix, SparseMatrix]) -> SparseMatrix:
    """Elementwise addition

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    B : DiagMatrix or SparseMatrix
        Diagonal matrix or sparse matrix

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
    if isinstance(B, DiagMatrix):
        B = B.as_sparse()
    if isinstance(A, SparseMatrix) and isinstance(B, SparseMatrix):
        return spsp_add(A, B)
    raise RuntimeError(
        "Elementwise addition between {} and {} is not "
        "supported.".format(type(A), type(B))
    )


def sp_power(A: SparseMatrix, scalar: Union[float, int]) -> SparseMatrix:
    """Take the power of each nonzero element and return a sparse matrix with
    the result.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    scalar : float or int
        Exponent

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    >>> row = torch.tensor([1, 0, 2])
    >>> col = torch.tensor([0, 3, 2])
    >>> val = torch.tensor([10, 20, 30])
    >>> A = create_from_coo(row, col, val)
    >>> A ** 2
    SparseMatrix(indices=tensor([[1, 0, 2],
            [0, 3, 2]]),
    values=tensor([100, 400, 900]),
    shape=(3, 4), nnz=3)
    """
    if isinstance(scalar, (float, int)):
        return val_like(A, A.val**scalar)

    raise RuntimeError(
        f"Raising a sparse matrix to exponent {type(scalar)} is not allowed."
    )


def sp_rpower(A: SparseMatrix, scalar: Union[float, int]):
    """Function for preventing raising a scalar to a sparse matrix exponent

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    scalar : float or int
        Scalar
    """
    raise RuntimeError(
        f"Raising {type(scalar)} to a sparse matrix component is not allowed."
    )


SparseMatrix.__add__ = sp_add
SparseMatrix.__radd__ = sp_add
SparseMatrix.__pow__ = sp_power
SparseMatrix.__rpow__ = sp_rpower
