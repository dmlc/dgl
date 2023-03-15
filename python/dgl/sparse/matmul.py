"""Matmul ops for SparseMatrix"""
# pylint: disable=invalid-name
from typing import Union

import torch

from .sparse_matrix import SparseMatrix

__all__ = ["spmm", "bspmm", "spspmm", "matmul"]


def spmm(A: SparseMatrix, X: torch.Tensor) -> torch.Tensor:
    """Multiplies a sparse matrix by a dense matrix, equivalent to ``A @ X``.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix of shape ``(L, M)`` with scalar values
    X : torch.Tensor
        Dense matrix of shape ``(M, N)`` or ``(M)``

    Returns
    -------
    torch.Tensor
        The dense matrix of shape ``(L, N)`` or ``(L)``

    Examples
    --------

    >>> indices = torch.tensor([[0, 1, 1], [1, 0, 1]])
    >>> val = torch.randn(indices.shape[1])
    >>> A = dglsp.spmatrix(indices, val)
    >>> X = torch.randn(2, 3)
    >>> result = dglsp.spmm(A, X)
    >>> type(result)
    <class 'torch.Tensor'>
    >>> result.shape
    torch.Size([2, 3])
    """
    assert isinstance(
        A, SparseMatrix
    ), f"Expect arg1 to be a SparseMatrix object, got {type(A)}."
    assert isinstance(
        X, torch.Tensor
    ), f"Expect arg2 to be a torch.Tensor, got {type(X)}."

    return torch.ops.dgl_sparse.spmm(A.c_sparse_matrix, X)


def bspmm(A: SparseMatrix, X: torch.Tensor) -> torch.Tensor:
    """Multiplies a sparse matrix by a dense matrix by batches, equivalent to
    ``A @ X``.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix of shape ``(L, M)`` with vector values of length ``K``
    X : torch.Tensor
        Dense matrix of shape ``(M, N, K)``

    Returns
    -------
    torch.Tensor
        Dense matrix of shape ``(L, N, K)``

    Examples
    --------

    >>> indices = torch.tensor([[0, 1, 1], [1, 0, 2]])
    >>> val = torch.randn(len(row), 2)
    >>> A = dglsp.spmatrix(indices, val, shape=(3, 3))
    >>> X = torch.randn(3, 3, 2)
    >>> result = dglsp.bspmm(A, X)
    >>> type(result)
    <class 'torch.Tensor'>
    >>> result.shape
    torch.Size([3, 3, 2])
    """
    assert isinstance(
        A, SparseMatrix
    ), f"Expect arg1 to be a SparseMatrix object, got {type(A)}."
    assert isinstance(
        X, torch.Tensor
    ), f"Expect arg2 to be a torch.Tensor, got {type(X)}."
    return spmm(A, X)


def spspmm(A: SparseMatrix, B: SparseMatrix) -> SparseMatrix:
    """Multiplies a sparse matrix by a sparse matrix, equivalent to ``A @ B``.

    The non-zero values of the two sparse matrices must be 1D.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix of shape ``(L, M)``
    B : SparseMatrix
        Sparse matrix of shape ``(M, N)``

    Returns
    -------
    SparseMatrix
        Sparse matrix of shape ``(L, N)``.

    Examples
    --------

    >>> indices1 = torch.tensor([[0, 1, 1], [1, 0, 1]])
    >>> val1 = torch.ones(len(row1))
    >>> A = dglsp.spmatrix(indices1, val1)
    >>> indices2 = torch.tensor([[0, 1, 1], [0, 2, 1]])
    >>> val2 = torch.ones(len(row2))
    >>> B = dglsp.spmatrix(indices2, val2)
    >>> dglsp.spspmm(A, B)
    SparseMatrix(indices=tensor([[0, 0, 1, 1, 1],
                                 [1, 2, 0, 1, 2]]),
                 values=tensor([1., 1., 1., 1., 1.]),
                 shape=(2, 3), nnz=5)
    """
    assert isinstance(
        A, SparseMatrix
    ), f"Expect A1 to be a SparseMatrix object, got {type(A)}."
    assert isinstance(
        B, SparseMatrix
    ), f"Expect A2 to be a SparseMatrix object, got {type(B)}."

    return SparseMatrix(
        torch.ops.dgl_sparse.spspmm(A.c_sparse_matrix, B.c_sparse_matrix)
    )


def matmul(
    A: Union[torch.Tensor, SparseMatrix], B: Union[torch.Tensor, SparseMatrix]
) -> Union[torch.Tensor, SparseMatrix]:
    """Multiplies two dense/sparse matrices, equivalent to ``A @ B``.

    This function does not support the case where :attr:`A` is a \
    ``torch.Tensor`` and :attr:`B` is a ``SparseMatrix``.

    * If both matrices are torch.Tensor, it calls \
        :func:`torch.matmul()`. The result is a dense matrix.

    * If both matrices are sparse, it calls :func:`dgl.sparse.spspmm`. The \
        result is a sparse matrix.

    * If :attr:`A` is sparse while :attr:`B` is dense, it calls \
        :func:`dgl.sparse.spmm`. The result is a dense matrix.

    * The operator supports batched sparse-dense matrix multiplication. In \
        this case, the sparse matrix :attr:`A` should have shape ``(L, M)``, \
        where the non-zero values have a batch dimension ``K``. The dense \
        matrix :attr:`B` should have shape ``(M, N, K)``. The output \
        is a dense matrix of shape ``(L, N, K)``.

    * Sparse-sparse matrix multiplication does not support batched computation.

    Parameters
    ----------
    A : torch.Tensor or SparseMatrix
        The first matrix.
    B : torch.Tensor or SparseMatrix
        The second matrix.

    Returns
    -------
    torch.Tensor or SparseMatrix
        The result matrix

    Examples
    --------

    Multiplies a diagonal matrix with a dense matrix.

    >>> val = torch.randn(3)
    >>> A = dglsp.diag(val)
    >>> B = torch.randn(3, 2)
    >>> result = dglsp.matmul(A, B)
    >>> type(result)
    <class 'torch.Tensor'>
    >>> result.shape
    torch.Size([3, 2])

    Multiplies a sparse matrix with a dense matrix.

    >>> indices = torch.tensor([[0, 1, 1], [1, 0, 1]])
    >>> val = torch.randn(indices.shape[1])
    >>> A = dglsp.spmatrix(indices, val)
    >>> X = torch.randn(2, 3)
    >>> result = dglsp.matmul(A, X)
    >>> type(result)
    <class 'torch.Tensor'>
    >>> result.shape
    torch.Size([2, 3])

    Multiplies a sparse matrix with a sparse matrix.

    >>> indices1 = torch.tensor([[0, 1, 1], [1, 0, 1]])
    >>> val1 = torch.ones(indices1.shape[1])
    >>> A = dglsp.spmatrix(indices1, val1)
    >>> indices2 = torch.tensor([[0, 1, 1], [0, 2, 1]])
    >>> val2 = torch.ones(indices2.shape[1])
    >>> B = dglsp.spmatrix(indices2, val2)
    >>> result = dglsp.matmul(A, B)
    >>> type(result)
    <class 'dgl.sparse.sparse_matrix.SparseMatrix'>
    >>> result.shape
    (2, 3)
    """
    assert isinstance(
        A, (torch.Tensor, SparseMatrix)
    ), f"Expect arg1 to be a torch.Tensor or SparseMatrix, got {type(A)}."
    assert isinstance(B, (torch.Tensor, SparseMatrix)), (
        f"Expect arg2 to be a torch Tensor or SparseMatrix"
        f"object, got {type(B)}."
    )
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return torch.matmul(A, B)
    assert not isinstance(A, torch.Tensor), (
        f"Expect arg2 to be a torch Tensor if arg 1 is torch Tensor, "
        f"got {type(B)}."
    )
    if isinstance(B, torch.Tensor):
        return spmm(A, B)
    return spspmm(A, B)


SparseMatrix.__matmul__ = matmul
