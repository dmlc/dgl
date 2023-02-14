"""Sampled Dense-Dense Matrix Multiplication (SDDMM) operator module."""
import torch

from .sparse_matrix import SparseMatrix

__all__ = ["sddmm", "bsddmm"]


# pylint: disable=invalid-name
def sddmm(A: SparseMatrix, X1: torch.Tensor, X2: torch.Tensor) -> SparseMatrix:
    r"""Sampled-Dense-Dense Matrix Multiplication (SDDMM).

    ``sddmm`` matrix-multiplies two dense matrices :attr:`X1` and :attr:`X2`,
    then elementwise-multiplies the result with sparse matrix :attr:`A` at the
    nonzero locations.

    Mathematically ``sddmm`` is formulated as:

    .. math::
        out = (X1 @ X2) * A

    In particular, :attr:`X1` and :attr:`X2` can be 1-D, then ``X1 @ X2``
    becomes the out-product of the two vectors (which results in a matrix).

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix of shape ``(L, N)``
    X1 : torch.Tensor
        Dense matrix of shape ``(L, M)`` or ``(L,)``
    X2 : torch.Tensor
        Dense matrix of shape ``(M, N)`` or ``(N,)``

    Returns
    -------
    SparseMatrix
        Sparse matrix of shape ``(L, N)``

    Examples
    --------

    >>> indices = torch.tensor([[1, 1, 2], [2, 3, 3]])
    >>> val = torch.arange(1, 4).float()
    >>> A = dglsp.spmatrix(indices, val, (3, 4))
    >>> X1 = torch.randn(3, 5)
    >>> X2 = torch.randn(5, 4)
    >>> dglsp.sddmm(A, X1, X2)
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 3, 3]]),
                 values=tensor([-1.6585, -3.9714, -0.5406]),
                 shape=(3, 4), nnz=3)
    """
    return SparseMatrix(torch.ops.dgl_sparse.sddmm(A.c_sparse_matrix, X1, X2))


# pylint: disable=invalid-name
def bsddmm(A: SparseMatrix, X1: torch.Tensor, X2: torch.Tensor) -> SparseMatrix:
    r"""Sampled-Dense-Dense Matrix Multiplication (SDDMM) by batches.

    ``sddmm`` matrix-multiplies two dense matrices :attr:`X1` and :attr:`X2`,
    then elementwise-multiplies the result with sparse matrix :attr:`A` at the
    nonzero locations.

    Mathematically ``sddmm`` is formulated as:

    .. math::
        out = (X1 @ X2) * A

    The batch dimension is the last dimension for input dense matrices. In
    particular, if the sparse matrix has scalar non-zero values, it will be
    broadcasted for bsddmm.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix of shape ``(L, N)`` with scalar values or vector values of
        length ``K``
    X1 : Tensor
        Dense matrix of shape ``(L, M, K)``
    X2 : Tensor
        Dense matrix of shape ``(M, N, K)``

    Returns
    -------
    SparseMatrix
        Sparse matrix of shape ``(L, N)`` with vector values of length ``K``

    Examples
    --------

    >>> indices = torch.tensor([[1, 1, 2], [2, 3, 3]])
    >>> val = torch.arange(1, 4).float()
    >>> A = dglsp.spmatrix(indices, val, (3, 4))
    >>> X1 = torch.arange(0, 3 * 5 * 2).view(3, 5, 2).float()
    >>> X2 = torch.arange(0, 5 * 4 * 2).view(5, 4, 2).float()
    >>> dglsp.bsddmm(A, X1, X2)
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 3, 3]]),
                 values=tensor([[1560., 1735.],
                                [3400., 3770.],
                                [8400., 9105.]]),
                 shape=(3, 4), nnz=3, val_size=(2,))
    """
    return sddmm(A, X1, X2)
