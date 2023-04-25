# pylint: disable=anomalous-backslash-in-string
"""DGL elementwise operator module."""
from typing import Union

from .sparse_matrix import SparseMatrix
from .utils import Scalar

__all__ = ["add", "sub", "mul", "div", "power"]


def add(A: SparseMatrix, B: SparseMatrix) -> SparseMatrix:
    r"""Elementwise addition for ``SparseMatrix``, equivalent to ``A + B``.

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
    >>> indices = torch.tensor([[1, 0, 2], [0, 1, 2]])
    >>> val = torch.tensor([10, 20, 30])
    >>> A = dglsp.spmatrix(indices, val)
    >>> B = dglsp.diag(torch.arange(1, 4))
    >>> dglsp.add(A, B)
    SparseMatrix(indices=tensor([[0, 0, 1, 1, 2],
                                 [0, 1, 0, 1, 2]]),
                 values=tensor([1, 20, 10,  2, 33]),
                 shape=(3, 3), nnz=5)
    """
    return A + B


def sub(A: SparseMatrix, B: SparseMatrix) -> SparseMatrix:
    r"""Elementwise subtraction for ``SparseMatrix``, equivalent to ``A - B``.

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
    >>> indices = torch.tensor([[1, 0, 2], [0, 1, 2]])
    >>> val = torch.tensor([10, 20, 30])
    >>> A = dglsp.spmatrix(indices, val)
    >>> B = dglsp.diag(torch.arange(1, 4))
    >>> dglsp.sub(A, B)
    SparseMatrix(indices=tensor([[0, 0, 1, 1, 2],
                                 [0, 1, 0, 1, 2]]),
                 values=tensor([-1, 20, 10, -2, 27]),
                 shape=(3, 3), nnz=5)
    """
    return A - B


def mul(
    A: Union[SparseMatrix, Scalar], B: Union[SparseMatrix, Scalar]
) -> SparseMatrix:
    r"""Elementwise multiplication for ``SparseMatrix``, equivalent to
    ``A * B``.

    If both :attr:`A` and :attr:`B` are sparse matrices, both of them should be
    diagonal matrices.

    Parameters
    ----------
    A : SparseMatrix or Scalar
        Sparse matrix or scalar value
    B : SparseMatrix or Scalar
        Sparse matrix or scalar value

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    >>> indices = torch.tensor([[1, 0, 2], [0, 3, 2]])
    >>> val = torch.tensor([10, 20, 30])
    >>> A = dglsp.spmatrix(indices, val)
    >>> dglsp.mul(A, 2)
    SparseMatrix(indices=tensor([[1, 0, 2],
                                 [0, 3, 2]]),
                 values=tensor([20, 40, 60]),
                 shape=(3, 4), nnz=3)

    >>> D = dglsp.diag(torch.arange(1, 4))
    >>> dglsp.mul(D, 2)
    SparseMatrix(indices=tensor([[0, 1, 2],
                                 [0, 1, 2]]),
                 values=tensor([2, 4, 6]),
                 shape=(3, 3), nnz=3)

    >>> D = dglsp.diag(torch.arange(1, 4))
    >>> dglsp.mul(D, D)
    SparseMatrix(indices=tensor([[0, 1, 2],
                                 [0, 1, 2]]),
                 values=tensor([1, 4, 9]),
                 shape=(3, 3), nnz=3)
    """
    return A * B


def div(A: SparseMatrix, B: Union[SparseMatrix, Scalar]) -> SparseMatrix:
    r"""Elementwise division for ``SparseMatrix``, equivalent to ``A / B``.

    If both :attr:`A` and :attr:`B` are sparse matrices, both of them should be
    diagonal matrices.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    B : SparseMatrix or Scalar
        Sparse matrix or scalar value

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    >>> A = dglsp.diag(torch.arange(1, 4))
    >>> B = dglsp.diag(torch.arange(10, 13))
    >>> dglsp.div(A, B)
    SparseMatrix(indices=tensor([[0, 1, 2],
                                 [0, 1, 2]]),
                 values=tensor([0.1000, 0.1818, 0.2500]),
                 shape=(3, 3), nnz=3)

    >>> A = dglsp.diag(torch.arange(1, 4))
    >>> dglsp.div(A, 2)
    SparseMatrix(indices=tensor([[0, 1, 2],
                                 [0, 1, 2]]),
                 values=tensor([0.5000, 1.0000, 1.5000]),
                 shape=(3, 3), nnz=3)

    >>> indices = torch.tensor([[1, 0, 2], [0, 3, 2]])
    >>> val = torch.tensor([1, 2, 3])
    >>> A = dglsp.spmatrix(indices, val, shape=(3, 4))
    >>> dglsp.div(A, 2)
    SparseMatrix(indices=tensor([[1, 0, 2],
                                 [0, 3, 2]]),
                 values=tensor([0.5000, 1.0000, 1.5000]),
                 shape=(3, 4), nnz=3)
    """
    return A / B


def power(A: SparseMatrix, scalar: Scalar) -> SparseMatrix:
    r"""Elementwise exponentiation ``SparseMatrix``, equivalent to
    ``A ** scalar``.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    scalar : Scalar
        Exponent

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    >>> indices = torch.tensor([[1, 0, 2], [0, 3, 2]])
    >>> val = torch.tensor([10, 20, 30])
    >>> A = dglsp.spmatrix(indices, val)
    >>> dglsp.power(A, 2)
    SparseMatrix(indices=tensor([[1, 0, 2],
                                 [0, 3, 2]]),
                 values=tensor([100, 400, 900]),
                 shape=(3, 4), nnz=3)

    >>> D = dglsp.diag(torch.arange(1, 4))
    >>> dglsp.power(D, 2)
    SparseMatrix(indices=tensor([[0, 1, 2],
                                 [0, 1, 2]]),
                 values=tensor([1, 4, 9]),
                 shape=(3, 3), nnz=3)
    """
    return A**scalar
