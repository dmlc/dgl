"""Sampled Dense-Dense Matrix Multiplication (SDDMM) operator module."""
import torch

from .sp_matrix import create_from_coo, SparseMatrix

__all__ = ["sddmm", "mock_bsddmm"]


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
    res = torch.sparse.sampled_addmm(
        A.adj.to_sparse_csr(), mat1, mat2
    ).to_sparse_coo()
    return SparseMatrix(A.row, A.col, res.values(), A.adj.shape)


def mock_bsddmm(
    A: SparseMatrix, mat1: torch.Tensor, mat2: torch.Tensor
) -> SparseMatrix:
    r"""Batched Sampled-Dense-Dense Matrix Multiplication (SDDMM).

    ``bsddmm`` conducts `sddmm` for each batch of the two dense matrices
    independently.

    In particular, :attr:``mat1`` and :attr:``mat2`` can be 2-D, which will be
    reshape as `(B, M, 1)` and `(B, 1, K)` in the computation.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix of shape `(M, N)`.
    mat1 : Tensor
        Dense matrix of shape `(B, M, K)` or `(B, M,)`
    mat2 : Tensor
        Dense matrix of shape `(B, K, N)` or `(B, K,)`

    Returns
    -------
    SparseMatrix
        Sparse matrix of shape `(M, N)` with non-zero values of `B` dimension.

    Examples
    --------

    >>> row = torch.tensor([1, 1, 2])
    >>> col = torch.tensor([2, 3, 3])
    >>> val = torch.arange(1, 4).float()
    >>> A = create_from_coo(row, col, val, (3, 4))
    >>> mat1 = torch.randn(2, 3, 5)
    >>> mat2 = torch.randn(2, 5, 4)
    >>> dgl.mock_sparse.mock_bsddmm(A, mat1, mat2)
    SparseMatrix(indices=tensor([[1, 1, 2],
            [2, 3, 3]]),
    values=tensor([[-0.6765, -0.4017],
            [ 3.3290,  6.9016],
            [ 4.8184,  5.8882]]),
    shape=(3, 4), nnz=3)
    """
    batch_mat1 = [mat1[i, ...] for i in range(mat1.shape[0])]
    batch_mat2 = [mat2[i, ...] for i in range(mat2.shape[0])]
    batch_ret = [sddmm(A, lhs, rhs) for lhs, rhs in zip(batch_mat1, batch_mat2)]
    return create_from_coo(
        row=A.row,
        col=A.col,
        val=torch.stack([sp_mat.val for sp_mat in batch_ret], dim=-1),
        shape=A.shape,
    )
