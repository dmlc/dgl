"""Softmax op for SparseMatrix"""
# pylint: disable=invalid-name

import torch

from .sparse_matrix import SparseMatrix

__all__ = ["softmax"]


def softmax(A: SparseMatrix) -> SparseMatrix:
    """Apply row-wise softmax to the non-zero entries of the sparse matrix.

    If :attr:`A.val` takes shape :attr:`(nnz, D)`, then the output matrix
    :attr:`A'` and :attr:`A'.val` take the same shape as :attr:`A` and
    :attr:`A.val`. :attr:`A'.val[:, i]` is calculated based on
    :attr:`A.val[:, i]`.

    Parameters
    ----------
    A : SparseMatrix
        The input sparse matrix

    Returns
    -------
    SparseMatrix
        The output sparse matrix

    Examples
    --------

    Case1: matrix with values of shape (nnz)

    >>> row = torch.tensor([0, 0, 1, 2])
    >>> col = torch.tensor([1, 2, 2, 0])
    >>> val = torch.ones(len(row))
    >>> A = create_from_coo(row, col, val)
    >>> softmax(A)
    TODO

    Case2: matrix with values of shape (nnz, D)

    >>> row = torch.tensor([0, 0, 1, 2])
    >>> col = torch.tensor([1, 2, 2, 0])
    >>> nnz = len(row)
    >>> val = torch.arange(nnz * 2).float().reshape(nnz, 2)
    >>> A = create_from_coo(row, col, val)
    >>> softmax(A)
    TODO
    """
    return SparseMatrix(torch.ops.dgl_sparse.softmax(A.c_sparse_matrix))
