"""Matmul ops for SparseMatrix"""
# pylint: disable=invalid-name
from typing import Union
import torch

from .sparse_matrix import SparseMatrix
from .diag_matrix import DiagMatrix

__all__ = ["spmm"]


def spmm(A: Union[SparseMatrix, DiagMatrix], X: torch.Tensor) -> torch.Tensor:
    """Multiply a sparse matrix by a dense matrix

    Parameters
    ----------
    A : SparseMatrix or DiagMatrix
        Sparse matrix of shape (N, M) with values of shape (nnz)
    X : torch.Tensor
        Dense tensor of shape (M, F) or (M)

    Returns
    -------
    torch.Tensor
        The result of multiplication

    Examples
    --------

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([1, 0, 1])
    >>> val = torch.randn(len(row))
    >>> A = create_from_coo(row, col, val)
    >>> X = torch.randn(2, 3)
    >>> result = A @ X
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
    assert (
        A.shape[1] == X.shape[0]
    ), f"Expect arg1.shape[1] == arg2.shape[0], got {A.shape[1]} and {X.shape[0]}"
    val_dim = len(A.val.shape)
    assert val_dim == 1, f"Expect arg1.val to be a 1D tensor, got {val_dim}D"
    val_dim = len(X.shape)
    assert val_dim <= 2, f"Expect arg2 to be a 1D/2D tensor, got {val_dim}D"

    if not isinstance(A, SparseMatrix):
        A = A.as_sparse()
    return torch.ops.dgl_sparse.spmm(A.c_sparse_matrix, X)


def mm_sp(
    A1: SparseMatrix, A2: Union[torch.Tensor, SparseMatrix, DiagMatrix]
) -> Union[torch.Tensor, SparseMatrix]:
    """Internal function for multiplying a sparse matrix by a dense/sparse/diagonal matrix

    Parameters
    ----------
    A1 : SparseMatrix
        Matrix of shape (N, M), with values of shape (nnz1)
    A2 : torch.Tensor, SparseMatrix, or DiagMatrix
        If A2 is a dense tensor, it can have shapes of (M, P) or (M, ).
        Otherwise it must have a shape of (M, P).

    Returns
    -------
    torch.Tensor or SparseMatrix
        The result of multiplication.

        * It is a dense torch tensor if :attr:`A2` is so.
        * It is a SparseMatrix object otherwise.

    Examples
    --------

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([1, 0, 1])
    >>> val = torch.randn(len(row))
    >>> A1 = create_from_coo(row, col, val)
    >>> A2 = torch.randn(2, 3)
    >>> result = A1 @ A2
    >>> print(type(result))
    <class 'torch.Tensor'>
    >>> print(result.shape)
    torch.Size([2, 3])
    """
    assert isinstance(
        A2, (torch.Tensor, SparseMatrix, DiagMatrix)
    ), f"Expect arg2 to be a torch Tensor, SparseMatrix, or DiagMatrix object, got {type(A2)}"

    if isinstance(A2, torch.Tensor):
        return spmm(A1, A2)
    else:
        raise NotImplementedError


SparseMatrix.__matmul__ = mm_sp
