"""DGL elementwise operators for sparse matrix module."""
from typing import Union

import torch

from .diag_matrix import DiagMatrix
from .sp_matrix import SparseMatrix

__all__ = ["sp_add", "sp_sub", "sp_mul", "sp_div", "sp_power"]


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
        assert A.shape == B.shape, (
            "The shape of sparse matrix A {} and"
            " B {} are expected to match".format(A.shape, B.shape)
        )
        C = (A.adj + B.adj).coalesce()
        return SparseMatrix(C.indices()[0], C.indices()[1], C.values(), C.shape)
    raise RuntimeError(
        "Elementwise addition between {} and {} is not "
        "supported.".format(type(A), type(B))
    )


def sp_sub(
    A: Union[SparseMatrix, DiagMatrix], B: Union[SparseMatrix, DiagMatrix]
) -> SparseMatrix:
    """Elementwise subtraction.

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
    Case 1: Subtract two sparse matrices

    >>> rowA = torch.tensor([1, 0, 2])
    >>> colA = torch.tensor([0, 3, 2])
    >>> valA = torch.tensor([10, 20, 30])
    >>> A = SparseMatrix(rowA, colA, valA, shape=(3, 4))
    >>> rowB = torch.tensor([1, 2, 0, 2, 1])
    >>> colB = torch.tensor([0, 2, 1, 3, 3])
    >>> valB = torch.tensor([1, 2, 3, 4, 5])
    >>> B = SparseMatrix(rowB, colB, valB, shape=(3 ,4))
    >>> A - B
    SparseMatrix(indices=tensor([[0, 0, 1, 1, 2, 2],
            [1, 3, 0, 3, 2, 3]]),
    values=tensor([-3, 20,  9, -5, 28, -4]),
    shape=(3, 4), nnz=6)

    Case 2: Subtract sparse matrix and diagonal matrix

    >>> D = diag(torch.arange(2, 5), shape=A.shape)
    >>> A - D
    SparseMatrix(indices=tensor([[0, 0, 1, 1, 2],
            [0, 3, 0, 1, 2]]),
    values=tensor([-2, 20, 10, -3, 26]),
    shape=(3, 4), nnz=5)
    """
    B = B.as_sparse() if isinstance(B, DiagMatrix) else B
    if isinstance(A, SparseMatrix) and isinstance(B, SparseMatrix):
        assert A.shape == B.shape, (
            "The shape of sparse matrix A {} and"
            " B {} are expected to match.".format(A.shape, B.shape)
        )
        C = A.adj - B.adj
        return SparseMatrix(C.indices()[0], C.indices()[1], C.values(), C.shape)
    raise RuntimeError(
        "Elementwise subtraction between {} and {} is not "
        "supported.".format(type(A), type(B))
    )


def sp_mul(
    A: Union[SparseMatrix, DiagMatrix, float],
    B: Union[SparseMatrix, DiagMatrix, float],
) -> SparseMatrix:
    """Elementwise multiplication.

    Parameters
    ----------
    A : SparseMatrix or DiagMatrix or scalar
        Sparse matrix or diagonal matrix or scalar value
    B : SparseMatrix or DiagMatrix or scalar
        Sparse matrix or diagonal matrix or scalar value

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    Case 1: Elementwise multiplication between two sparse matrices

    >>> rowA = torch.tensor([1, 0, 2])
    >>> colA = torch.tensor([0, 3, 2])
    >>> valA = torch.tensor([10, 20, 30])
    >>> A = SparseMatrix(rowA, colA, valA, shape=(3, 4))
    >>> rowB = torch.tensor([1, 2, 0, 2, 1])
    >>> colB = torch.tensor([0, 2, 1, 3, 3])
    >>> valB = torch.tensor([1, 2, 3, 4, 5])
    >>> B = SparseMatrix(rowB, colB, valB, shape=(3 ,4))
    >>> A * B
    SparseMatrix(indices=tensor([[1, 2],
            [0, 2]]),
    values=tensor([10, 60]),
    shape=(3, 4), nnz=2)

    Case 2: Elementwise multiplication between sparse matrix and scalar value

    >>> v_scalar = 2.5
    >>> A * v_scalar
    SparseMatrix(indices=tensor([[0, 1, 2],
            [3, 0, 2]]),
    values=tensor([50., 25., 75.]),
    shape=(3, 4), nnz=3)

    Case 3: Elementwise multiplication between sparse and diagonal matrix

    >>> D = diag(torch.arange(2, 5), shape=A.shape)
    >>> A * D
    SparseMatrix(indices=tensor([[2],
            [2]]),
    values=tensor([120]),
    shape=(3, 4), nnz=1)
    """
    B = B.as_sparse() if isinstance(B, DiagMatrix) else B
    if isinstance(A, SparseMatrix) and isinstance(B, SparseMatrix):
        assert A.shape == B.shape, (
            "The shape of sparse matrix A {} and"
            " B {} are expected to match.".format(A.shape, B.shape)
        )
    A = A.adj if isinstance(A, SparseMatrix) else A
    B = B.adj if isinstance(B, SparseMatrix) else B
    C = A * B
    return SparseMatrix(C.indices()[0], C.indices()[1], C.values(), C.shape)


def sp_div(
    A: Union[SparseMatrix, DiagMatrix],
    B: Union[SparseMatrix, DiagMatrix, float],
) -> SparseMatrix:
    """Elementwise division.

    Parameters
    ----------
    A : SparseMatrix or DiagMatrix
        Sparse matrix or diagonal matrix
    B : SparseMatrix or DiagMatrix or scalar
        Sparse matrix or diagonal matrix or scalar value.

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    Case 1: Elementwise division between two matrices of same sparsity (matrices
            with different sparsity is not supported)

    >>> rowA = torch.tensor([1, 0, 2, 7, 1])
    >>> colA = torch.tensor([0, 49, 2, 1, 7])
    >>> valA = torch.tensor([10, 20, 30, 40, 50])
    >>> A = SparseMatrix(rowA, colA, valA, shape=(10, 50))
    >>> w = torch.arange(1, len(rowA)+1)
    >>> A/A(w)
    SparseMatrix(indices=tensor([[ 0,  1,  1,  2,  7],
            [49,  0,  7,  2,  1]]),
    values=tensor([20.0000,  5.0000, 16.6667,  7.5000,  8.0000]),
    shape=(10, 50), nnz=5)

    Case 2: Elementwise multiplication between sparse matrix and scalar value

    >>> v_scalar = 2.5
    >>> A / v_scalar
    SparseMatrix(indices=tensor([[ 0,  1,  1,  2,  7],
            [49,  0,  7,  2,  1]]),
    values=tensor([ 8.,  4., 20., 12., 16.]),
    shape=(10, 50), nnz=5)
    """
    B = B.as_sparse() if isinstance(B, DiagMatrix) else B
    if isinstance(A, SparseMatrix) and isinstance(B, SparseMatrix):
        # same sparsity structure
        if torch.equal(A.indices("COO"), B.indices("COO")):
            return SparseMatrix(A.row, A.col, A.val / B.val, A.shape)
        raise ValueError(
            "Division between matrices of different sparsity is not supported"
        )
    C = A.adj / B
    return SparseMatrix(C.indices()[0], C.indices()[1], C.values(), C.shape)


def sp_rdiv(A: float, B: Union[SparseMatrix, DiagMatrix]):
    """Elementwise division.

    Parameters
    ----------
    A : scalar
        scalar value
    B : SparseMatrix or DiagMatrix
        Sparse matrix or diagonal matrix
    """
    raise RuntimeError(
        "Elementwise division between {} and {} is not "
        "supported.".format(type(A), type(B))
    )


def sp_power(A: SparseMatrix, B: float) -> SparseMatrix:
    """Elementwise power operation.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    B : scalar
        scalar value.

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    >>> rowA = torch.tensor([1, 0, 2, 7, 1])
    >>> colA = torch.tensor([0, 49, 2, 1, 7])
    >>> valA = torch.tensor([10, 20, 30, 40, 50])
    >>> A = SparseMatrix(rowA, colA, valA, shape=(10, 50))
    >>> pow(A, 2.5)
    SparseMatrix(indices=tensor([[ 0,  1,  1,  2,  7],
            [49,  0,  7,  2,  1]]),
    values=tensor([ 1788.8544,   316.2278, 17677.6699,  4929.5029, 10119.2881]),
    shape=(10, 50), nnz=5)
    """
    if isinstance(B, SparseMatrix):
        raise RuntimeError(
            "Power operation between two sparse matrices is not supported"
        )
    return SparseMatrix(A.row, A.col, torch.pow(A.val, B), A.shape)


def sp_rpower(A: float, B: SparseMatrix) -> SparseMatrix:
    """Elementwise power operation.

    Parameters
    ----------
    A : scalar
        scalar value.
    B : SparseMatrix
        Sparse matrix.
    """
    raise RuntimeError(
        "Power operation between {} and {} is not "
        "supported.".format(type(A), type(B))
    )


SparseMatrix.__add__ = sp_add
SparseMatrix.__radd__ = sp_add
SparseMatrix.__sub__ = sp_sub
SparseMatrix.__rsub__ = sp_sub
SparseMatrix.__mul__ = sp_mul
SparseMatrix.__rmul__ = sp_mul
SparseMatrix.__truediv__ = sp_div
SparseMatrix.__rtruediv__ = sp_rdiv
SparseMatrix.__pow__ = sp_power
SparseMatrix.__rpow__ = sp_rpower
