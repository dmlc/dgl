"""Softmax op for SparseMatrix"""
# pylint: disable=invalid-name, W0622

import torch

from .sparse_matrix import SparseMatrix

__all__ = ["softmax"]


def softmax(input: SparseMatrix) -> SparseMatrix:
    """Applies row-wise softmax to the non-zero elements of the sparse matrix.

    Equivalently, applies softmax to the non-zero elements of the sparse
    matrix along the column (``dim=1``) dimension.

    If :attr:`input.val` takes shape ``(nnz, D)``, then the output matrix
    :attr:`output` and :attr:`output.val` take the same shape as :attr:`input`
    and :attr:`input.val`. :attr:`output.val[:, i]` is calculated based on
    :attr:`input.val[:, i]`.

    Parameters
    ----------
    input : SparseMatrix
        The input sparse matrix

    Returns
    -------
    SparseMatrix
        The output sparse matrix

    Examples
    --------

    Case1: matrix with values of shape (nnz)

    >>> indices = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]])
    >>> nnz = len(row)
    >>> val = torch.arange(nnz).float()
    >>> A = dglsp.spmatrix(indices, val)
    >>> dglsp.softmax(A)
    SparseMatrix(indices=tensor([[0, 0, 1, 2],
                                 [1, 2, 2, 0]]),
                 values=tensor([0.2689, 0.7311, 1.0000, 1.0000]),
                 shape=(3, 3), nnz=4)

    Case2: matrix with values of shape (nnz, D)

    >>> indices = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]])
    >>> val = torch.tensor([[0., 7.], [1., 3.], [2., 2.], [3., 1.]])
    >>> A = dglsp.spmatrix(indices, val)
    >>> dglsp.softmax(A)
    SparseMatrix(indices=tensor([[0, 0, 1, 2],
                                 [1, 2, 2, 0]]),
                 values=tensor([[0.2689, 0.9820],
                                [0.7311, 0.0180],
                                [1.0000, 1.0000],
                                [1.0000, 1.0000]]),
                 shape=(3, 3), nnz=4, val_size=(2,))
    """
    return SparseMatrix(torch.ops.dgl_sparse.softmax(input.c_sparse_matrix))


SparseMatrix.softmax = softmax
