"""Sampled Dense-Dense Matrix Multiplication (SDDMM) operator module."""
import torch

from .sp_matrix import SparseMatrix

__all__ = ["sddmm"]


def sddmm(
    A: SparseMatrix, mat1: torch.Tensor, mat2: torch.Tensor
) -> SparseMatrix:
    r"""Sampled-Dense-Dense Matrix Multiplication (SDDMM).

    ``sddmm`` multiplies two dense matrices :attr:``mat1`` and :attr:``mat2``
    at the nonzero locations of sparse matrix :attr:``A``. Values of :attr:``A``
    is added to the resulting matrix.

    Mathematically ``sddmm`` is formulated as:

    .. math::
        out = (mat1 @ mat2) * spy(A) + A

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

    >>> row = torch.Tensor([1, 1, 2])
    >>> col = torch.Tensor([2, 3, 3])
    >>> val = torch.arange(1, 4).float()
    >>> A = SparseMatrix(row, col, val, (3, 4))
    >>> mat1 = torch.randn(3, 5)
    >>> mat2 = torch.randn(5, 4)
    >>> dgl.mock_sparse.sddmm(A, mat1, mat2)
    SparseMatrix(indices=tensor([[1, 1, 2],
            [2, 3, 3]]),
    values=tensor([1.8035, 2.3375, 3.1255]),
    shape=(3, 4), nnz=3)
    """
    assert A.val.dim() == 1, (
        f"Nonzero elements have values of shape ({A.val.shape[1]}). Expects "
        "scalar values. "
    )
    # PyTorch's sddmm operator only supports CSR format.
    res = torch.sparse.sampled_addmm(A.adj.to_sparse_csr(), mat1, mat2)
    return SparseMatrix(A.row, A.col, res.values(), A.adj.shape)
