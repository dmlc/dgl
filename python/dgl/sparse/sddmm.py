"""Sampled Dense-Dense Matrix Multiplication (SDDMM) operator module."""
import torch

from .sparse_matrix import SparseMatrix

__all__ = ["sddmm", "bsddmm"]


def sddmm(
    A: SparseMatrix, mat1: torch.Tensor, mat2: torch.Tensor
) -> SparseMatrix:
    r"""Sampled-Dense-Dense Matrix Multiplication (SDDMM).

    ``sddmm`` matrix-multiplies two dense matrices :attr:`mat1` and :attr:`mat2`
    , then elementwise-multiplies the result with sparse matrix :attr:`A` at the
    nonzero locations.

    Mathematically ``sddmm`` is formulated as:

    .. math::
        out = (mat1 @ mat2) * A

    In particular, :attr:`mat1` and :attr:`mat2` can be 1-D, then ``mat1 @
    mat2`` becomes the out-product of the two vector (which results in a
    matrix).

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix of shape ``(M, N)``.
    mat1 : Tensor
        Dense matrix of shape ``(M, K)`` or ``(M,)``
    mat2 : Tensor
        Dense matrix of shape ``(K, N)`` or ``(N,)``

    Returns
    -------
    SparseMatrix
        Sparse matrix of shape ``(M, N)``.

    Examples
    --------

    >>> row = torch.tensor([1, 1, 2])
    >>> col = torch.tensor([2, 3, 3])
    >>> val = torch.arange(1, 4).float()
    >>> A = from_coo(row, col, val, (3, 4))
    >>> mat1 = torch.randn(3, 5)
    >>> mat2 = torch.randn(5, 4)
    >>> dgl.sparse.sddmm(A, mat1, mat2)
    SparseMatrix(indices=tensor([[1, 1, 2],
            [2, 3, 3]]),
    values=tensor([ 1.3097, -1.0977,  1.6953]),
    shape=(3, 4), nnz=3)
    """
    return SparseMatrix(
        torch.ops.dgl_sparse.sddmm(A.c_sparse_matrix, mat1, mat2)
    )


def bsddmm(
    A: SparseMatrix, mat1: torch.Tensor, mat2: torch.Tensor
) -> SparseMatrix:
    r"""Sampled-Dense-Dense Matrix Multiplication (SDDMM) by batches.

    ``sddmm`` multiplies two dense matrices :attr:`mat1` and :attr:`mat2`
    at the nonzero locations of sparse matrix :attr:`A`. Values of :attr:`A`
    is not considered during the computation.

    Mathematically ``sddmm`` is formulated as:

    .. math::
        out = (mat1 @ mat2) * A

    The batch dimension is the last dimension for input matrices. In particular,
    if the sparse matrix has scalar non-zero values, it will be broadcasted
    for bsddmm.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix of shape ``(M, N)`` or ``(M, N, B)``.
    mat1 : Tensor
        Dense matrix of shape ``(M, K, B)``
    mat2 : Tensor
        Dense matrix of shape ``(K, N, B)``

    Returns
    -------
    SparseMatrix
        Sparse matrix of shape ``(M, N, B)``.

    Examples
    --------

    >>> row = torch.tensor([1, 1, 2])
    >>> col = torch.tensor([2, 3, 3])
    >>> val = torch.arange(1, 4).float()
    >>> A = from_coo(row, col, val, (3, 4))
    >>> mat1 = torch.arange(0, 3 * 5 * 2).view(3, 5, 2).float()
    >>> mat2 = torch.arange(0, 5 * 4 * 2).view(5, 4, 2).float()
    >>> dgl.sparse.bsddmm(A, mat1, mat2)
    SparseMatrix(indices=tensor([[1, 1, 2],
            [2, 3, 3]]),
    values=tensor([[1560., 1735.],
            [3400., 3770.],
            [8400., 9105.]]),
    shape=(3, 4), nnz=3)
    """
    return sddmm(A, mat1, mat2)
