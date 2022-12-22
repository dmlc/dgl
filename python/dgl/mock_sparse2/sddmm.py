"""Sampled Dense-Dense Matrix Multiplication (SDDMM) operator module."""
import torch

from .sparse_matrix import SparseMatrix

__all__ = ["sddmm"]


def sddmm(
    A: SparseMatrix, mat1: torch.Tensor, mat2: torch.Tensor
) -> SparseMatrix:
    r"""Sampled-Dense-Dense Matrix Multiplication (SDDMM).

    ``sddmm`` multiplies two dense matrices :attr:``mat1`` and :attr:``mat2``
    at the nonzero locations of sparse matrix :attr:``A``. Values of :attr:``A``
    is not considered during the computation.

    Mathematically ``sddmm`` is formulated as:

    .. math::
        out = (mat1 @ mat2) * A

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix of shape `(M, N)`.
    mat1 : Tensor
        Dense matrix of shape `(M, K)`
    mat2 : Tensor
        Dense matrix of shape `(K, N)`

    Returns
    -------
    SparseMatrix
        Sparse matrix of shape `(M, N)`.

    Examples
    --------

    >>> row = torch.tensor([1, 1, 2])
    >>> col = torch.tensor([2, 3, 3])
    >>> val = torch.arange(1, 4).float()
    >>> A = create_from_coo(row, col, val, (3, 4))
    >>> mat1 = torch.randn(3, 5)
    >>> mat2 = torch.randn(5, 4)
    >>> dgl.mock_sparse.sddmm(A, mat1, mat2)
    SparseMatrix(indices=tensor([[1, 1, 2],
            [2, 3, 3]]),
    values=tensor([ 1.3097, -1.0977,  1.6953]),
    shape=(3, 4), nnz=3)
    """
    return SparseMatrix(
        torch.ops.dgl_sparse.sddmm(A.c_sparse_matrix, mat1, mat2)
    )
