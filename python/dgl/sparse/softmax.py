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
    >>> nnz = len(row)
    >>> val = torch.arange(nnz).float()
    >>> A = from_coo(row, col, val)
    >>> softmax(A)
    SparseMatrix(indices=tensor([[0, 0, 1, 2],
        [1, 2, 2, 0]]),
    values=tensor([0.2689, 0.7311, 1.0000, 1.0000]),
    shape=(3, 3), nnz=4)

    Case2: matrix with values of shape (nnz, D)

    >>> val = torch.tensor([[0., 7.], [1., 3.], [2., 2.], [3., 1.]])
    >>> A = from_coo(row, col, val)
    >>> softmax(A)
    SparseMatrix(indices=tensor([[0, 0, 1, 2],
        [1, 2, 2, 0]]),
    values=tensor([[0.2689, 0.9820],
        [0.7311, 0.0180],
        [1.0000, 1.0000],
        [1.0000, 1.0000]]),
    shape=(3, 3), nnz=4)
    """
    return SparseMatrix(torch.ops.dgl_sparse.softmax(A.c_sparse_matrix))


SparseMatrix.softmax = softmax
