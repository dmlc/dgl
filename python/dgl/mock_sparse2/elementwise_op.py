"""DGL elementwise operator module."""
from typing import Union

from .diag_matrix import DiagMatrix
from .elementwise_op_diag import diag_add
from .elementwise_op_sp import sp_add
from .sparse_matrix import SparseMatrix

__all__ = ["add", "power"]


def add(
    A: Union[SparseMatrix, DiagMatrix], B: Union[SparseMatrix, DiagMatrix]
) -> Union[SparseMatrix, DiagMatrix]:
    """Elementwise addition"""
    if isinstance(A, DiagMatrix) and isinstance(B, DiagMatrix):
        return diag_add(A, B)
    return sp_add(A, B)


def power(
    A: Union[SparseMatrix, DiagMatrix], scalar: Union[float, int]
) -> Union[SparseMatrix, DiagMatrix]:
    """Take the power of each nonzero element and return a matrix with
    the result.

    Parameters
    ----------
    A : SparseMatrix or DiagMatrix
        Sparse matrix or diagonal matrix
    scalar : float or int
        Exponent

    Returns
    -------
    SparseMatrix or DiagMatrix
        Sparse matrix or diagonal matrix, same type as A

    Examples
    --------

    >>> row = torch.tensor([1, 0, 2])
    >>> col = torch.tensor([0, 3, 2])
    >>> val = torch.tensor([10, 20, 30])
    >>> A = create_from_coo(row, col, val)
    >>> power(A, 2)
    SparseMatrix(indices=tensor([[1, 0, 2],
            [0, 3, 2]]),
    values=tensor([100, 400, 900]),
    shape=(3, 4), nnz=3)

    >>> D = diag(torch.arange(1, 4))
    >>> power(D, 2)
    DiagMatrix(val=tensor([1, 4, 9]),
               shape=(3, 3))
    """
    return A**scalar
