# pylint: disable=anomalous-backslash-in-string
"""DGL elementwise operator module."""
from typing import Union

from .diag_matrix import DiagMatrix
from .sparse_matrix import SparseMatrix

__all__ = ["add", "power"]


def add(
    A: Union[DiagMatrix, SparseMatrix], B: Union[DiagMatrix, SparseMatrix]
) -> Union[DiagMatrix, SparseMatrix]:
    """Elementwise additions for `DiagMatrix` and `SparseMatrix`.

    The supported combinations are shown as follow.
    +--------------+------------+--------------+--------+
    |     A \ B    | DiagMatrix | SparseMatrix | scalar |
    +--------------+------------+--------------+--------+
    |  DiagMatrix  |     ✅     |      ✅      |   🚫   |
    +--------------+------------+--------------+--------+
    | SparseMatrix |     ✅     |      ✅      |   🚫   |
    +--------------+------------+--------------+--------+
    |    scalar    |     🚫     |      🚫      |   🚫   |
    +--------------+------------+--------------+--------+

    Parameters
    ----------
    A : DiagMatrix or SparseMatrix
        Diagonal matrix or sparse matrix
    B : DiagMatrix or SparseMatrix
        Diagonal matrix or sparse matrix

    Returns
    -------
    DiagMatrix or SparseMatrix
        Diagonal matrix if both :attr:`A` and :attr:`B` are diagonal matrices,
        sparse matrix otherwise

    Examples
    --------
    >>> row = torch.tensor([1, 0, 2])
    >>> col = torch.tensor([0, 1, 2])
    >>> val = torch.tensor([10, 20, 30])
    >>> A = from_coo(row, col, val)
    >>> B = diag(torch.arange(1, 4))
    >>> A + B
    SparseMatrix(indices=tensor([[0, 0, 1, 1, 2],
                                 [0, 1, 0, 1, 2]]),
    values=tensor([ 1, 20, 10,  2, 33]),
    shape=(3, 3), nnz=5)
    """
    return A + B


def sub(A: Union[DiagMatrix], B: Union[DiagMatrix]) -> Union[DiagMatrix]:
    """Elementwise subtraction for `DiagMatrix` and `SparseMatrix`.

    The supported combinations are shown as follow.
    +--------------+------------+--------------+--------+
    |     A \ B    | DiagMatrix | SparseMatrix | scalar |
    +--------------+------------+--------------+--------+
    |  DiagMatrix  |     ✅     |      🚫      |   🚫   |
    +--------------+------------+--------------+--------+
    | SparseMatrix |     🚫     |      🚫      |   🚫   |
    +--------------+------------+--------------+--------+
    |    scalar    |     🚫     |      🚫      |   🚫   |
    +--------------+------------+--------------+--------+

    Parameters
    ----------
    A : DiagMatrix
        Diagonal matrix
    B : DiagMatrix
        Diagonal matrix

    Returns
    -------
    DiagMatrix
        Diagonal matrix

    Examples
    --------
    >>> A = diag(torch.arange(1, 4))
    >>> B = diag(torch.arange(10, 13))
    >>> A - B
    DiagMatrix(val=tensor([-9, -9, -9]),
    shape=(3, 3))
    """
    return A - B


def power(
    A: Union[SparseMatrix, DiagMatrix], scalar: Union[float, int]
) -> Union[SparseMatrix, DiagMatrix]:
    """Elementwise exponentiation for `DiagMatrix` and `SparseMatrix`.

    The supported combinations are shown as follow.
    +--------------+------------+--------------+--------+
    |     A \ B    | DiagMatrix | SparseMatrix | scalar |
    +--------------+------------+--------------+--------+
    |  DiagMatrix  |     🚫     |      🚫      |   ✅   |
    +--------------+------------+--------------+--------+
    | SparseMatrix |     🚫     |      🚫      |   ✅   |
    +--------------+------------+--------------+--------+
    |    scalar    |     🚫     |      🚫      |   🚫   |
    +--------------+------------+--------------+--------+

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
    >>> A = from_coo(row, col, val)
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
