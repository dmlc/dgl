"""Matmul ops for SparseMatrix"""
# pylint: disable=invalid-name
from typing import Union

import torch

from .diag_matrix import diag, DiagMatrix

from .sparse_matrix import SparseMatrix

__all__ = ["spmm", "spspmm", "mm"]


def spmm(A: Union[SparseMatrix, DiagMatrix], X: torch.Tensor) -> torch.Tensor:
    """Multiply a sparse matrix by a dense matrix.

    Parameters
    ----------
    A : SparseMatrix or DiagMatrix
        Sparse matrix of shape (N, M) with values of shape (nnz)
    X : torch.Tensor
        Dense tensor of shape (M, F) or (M)

    Returns
    -------
    torch.Tensor
        The multiplication result of shape (N, F) or (N)

    Examples
    --------

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([1, 0, 1])
    >>> val = torch.randn(len(row))
    >>> A = create_from_coo(row, col, val)
    >>> X = torch.randn(2, 3)
    >>> result = dgl.sparse.spmm(A, X)
    >>> print(type(result))
    <class 'torch.Tensor'>
    >>> print(result.shape)
    torch.Size([2, 3])
    """
    assert isinstance(
        A, (SparseMatrix, DiagMatrix)
    ), f"Expect arg1 to be a SparseMatrix or DiagMatrix object, got {type(A)}"
    assert isinstance(
        X, torch.Tensor
    ), f"Expect arg2 to be a torch.Tensor, got {type(X)}"

    # The input is a DiagMatrix. Cast it to SparseMatrix
    if not isinstance(A, SparseMatrix):
        A = A.as_sparse()
    return torch.ops.dgl_sparse.spmm(A.c_sparse_matrix, X)


def _diag_diag_mm(A1: DiagMatrix, A2: DiagMatrix) -> DiagMatrix:
    """Internal function for multiplying a diagonal matrix by a diagonal matrix

    Parameters
    ----------
    A1 : DiagMatrix
        Matrix of shape (N, M), with values of shape (nnz1)
    A2 : DiagMatrix
        Matrix of shape (M, P), with values of shape (nnz2)

    Returns
    -------
    DiagMatrix
        The result of multiplication.
    """
    M, N = A1.shape
    N, P = A2.shape
    common_diag_len = min(M, N, P)
    new_diag_len = min(M, P)
    diag_val = torch.zeros(new_diag_len)
    diag_val[:common_diag_len] = (
        A1.val[:common_diag_len] * A2.val[:common_diag_len]
    )
    return diag(diag_val.to(A1.device), (M, P))


def spspmm(
    A1: Union[SparseMatrix, DiagMatrix], A2: Union[SparseMatrix, DiagMatrix]
) -> Union[SparseMatrix, DiagMatrix]:
    """Multiply a sparse matrix by a sparse matrix. The non-zero values of the
    two sparse matrices must be 1D.

    Parameters
    ----------
    A1 : SparseMatrix or DiagMatrix
        Sparse matrix of shape (N, M) with values of shape (nnz)
    A2 : SparseMatrix or DiagMatrix
        Sparse matrix of shape (M, P) with values of shape (nnz)

    Returns
    -------
    SparseMatrix or DiagMatrix
        The result of multiplication. It is a DiagMatrix object if both matrices
        are DiagMatrix objects. It is a SparseMatrix object otherwise.

    Examples
    --------

    >>> row1 = torch.tensor([0, 1, 1])
    >>> col1 = torch.tensor([1, 0, 1])
    >>> val1 = torch.ones(len(row1))
    >>> A1 = create_from_coo(row1, col1, val1)

    >>> row2 = torch.tensor([0, 1, 1])
    >>> col2 = torch.tensor([0, 2, 1])
    >>> val2 = torch.ones(len(row2))
    >>> A2 = create_from_coo(row2, col2, val2)
    >>> result = dgl.sparse.spspmm(A1, A2)
    >>> print(result)
    SparseMatrix(indices=tensor([[0, 0, 1, 1, 1],
                                 [1, 2, 0, 1, 2]]),
                 values=tensor([1., 1., 1., 1., 1.]),
                 shape=(2, 3), nnz=5)
    """
    assert isinstance(
        A1, (SparseMatrix, DiagMatrix)
    ), f"Expect A1 to be a SparseMatrix or DiagMatrix object, got {type(A1)}"
    assert isinstance(
        A2, (SparseMatrix, DiagMatrix)
    ), f"Expect A2 to be a SparseMatrix or DiagMatrix object, got {type(A2)}"

    if isinstance(A1, DiagMatrix) and isinstance(A2, DiagMatrix):
        return _diag_diag_mm(A1, A2)
    if isinstance(A1, DiagMatrix):
        A1 = A1.as_sparse()
    if isinstance(A2, DiagMatrix):
        A2 = A2.as_sparse()
    return SparseMatrix(
        torch.ops.dgl_sparse.spspmm(A1.c_sparse_matrix, A2.c_sparse_matrix)
    )


def mm(
    A1: Union[SparseMatrix, DiagMatrix],
    A2: Union[torch.Tensor, SparseMatrix, DiagMatrix],
) -> Union[torch.Tensor, SparseMatrix, DiagMatrix]:
    """Multiply a sparse/diagonal matrix by a dense/sparse/diagonal matrix.
    If an input is a SparseMatrix or DiagMatrix, its non-zero values should
    be 1-D.

    Parameters
    ----------
    A1 : SparseMatrix or DiagMatrix
        Matrix of shape (N, M), with values of shape (nnz1)
    A2 : torch.Tensor, SparseMatrix, or DiagMatrix
        Matrix of shape (M, P). If it is a SparseMatrix or DiagMatrix,
        it should have values of shape (nnz2).

    Returns
    -------
    torch.Tensor or DiagMatrix or SparseMatrix
        The result of multiplication of shape (N, P)

        * It is a dense torch tensor if :attr:`A2` is so.
        * It is a DiagMatrix object if both :attr:`A1` and :attr:`A2` are so.
        * It is a SparseMatrix object otherwise.

    Examples
    --------

    >>> val = torch.randn(3)
    >>> A1 = diag(val)
    >>> A2 = torch.randn(3, 2)
    >>> result = dgl.sparse.mm(A1, A2)
    >>> print(type(result))
    <class 'torch.Tensor'>
    >>> print(result.shape)
    torch.Size([3, 2])
    """
    assert isinstance(
        A1, (SparseMatrix, DiagMatrix)
    ), f"Expect arg1 to be a SparseMatrix, or DiagMatrix object, got {type(A1)}."
    assert isinstance(A2, (torch.Tensor, SparseMatrix, DiagMatrix)), (
        f"Expect arg2 to be a torch Tensor, SparseMatrix, or DiagMatrix"
        f"object, got {type(A2)}."
    )
    if isinstance(A2, torch.Tensor):
        return spmm(A1, A2)
    if isinstance(A1, DiagMatrix) and isinstance(A2, DiagMatrix):
        return _diag_diag_mm(A1, A2)
    return spspmm(A1, A2)


SparseMatrix.__matmul__ = mm
DiagMatrix.__matmul__ = mm
