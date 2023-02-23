"""Matmul ops for SparseMatrix"""
# pylint: disable=invalid-name
from typing import Union

import torch

from .diag_matrix import diag, DiagMatrix

from .sparse_matrix import SparseMatrix, val_like

__all__ = ["spmm", "bspmm", "spspmm", "matmul"]


def spmm(A: Union[SparseMatrix, DiagMatrix], X: torch.Tensor) -> torch.Tensor:
    """Multiplies a sparse matrix by a dense matrix, equivalent to ``A @ X``.

    Parameters
    ----------
    A : SparseMatrix or DiagMatrix
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
    >>> val = torch.randn(len(row))
    >>> A = dglsp.spmatrix(indices, val)
    >>> X = torch.randn(2, 3)
    >>> result = dglsp.spmm(A, X)
    >>> type(result)
    <class 'torch.Tensor'>
    >>> result.shape
    torch.Size([2, 3])
    """
    assert isinstance(
        A, (SparseMatrix, DiagMatrix)
    ), f"Expect arg1 to be a SparseMatrix or DiagMatrix object, got {type(A)}."
    assert isinstance(
        X, torch.Tensor
    ), f"Expect arg2 to be a torch.Tensor, got {type(X)}."

    # The input is a DiagMatrix. Cast it to SparseMatrix
    if not isinstance(A, SparseMatrix):
        A = A.to_sparse()
    return torch.ops.dgl_sparse.spmm(A.c_sparse_matrix, X)


def bspmm(A: Union[SparseMatrix, DiagMatrix], X: torch.Tensor) -> torch.Tensor:
    """Multiplies a sparse matrix by a dense matrix by batches, equivalent to
    ``A @ X``.

    Parameters
    ----------
    A : SparseMatrix or DiagMatrix
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
        A, (SparseMatrix, DiagMatrix)
    ), f"Expect arg1 to be a SparseMatrix or DiagMatrix object, got {type(A)}."
    assert isinstance(
        X, torch.Tensor
    ), f"Expect arg2 to be a torch.Tensor, got {type(X)}."
    return spmm(A, X)


def _diag_diag_mm(A: DiagMatrix, B: DiagMatrix) -> DiagMatrix:
    """Internal function for multiplying a diagonal matrix by a diagonal matrix.

    Parameters
    ----------
    A : DiagMatrix
        Diagonal matrix of shape ``(L, M)``
    B : DiagMatrix
        Diagonal matrix of shape ``(M, N)``

    Returns
    -------
    DiagMatrix
        Diagonal matrix of shape ``(L, N)``
    """
    M, N = A.shape
    N, P = B.shape
    common_diag_len = min(M, N, P)
    new_diag_len = min(M, P)
    diag_val = torch.zeros(new_diag_len)
    diag_val[:common_diag_len] = (
        A.val[:common_diag_len] * B.val[:common_diag_len]
    )
    return diag(diag_val.to(A.device), (M, P))


def _sparse_diag_mm(A, D):
    """Internal function for multiplying a sparse matrix by a diagonal matrix.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix of shape ``(L, M)``
    D : DiagMatrix
        Diagonal matrix of shape ``(M, N)``

    Returns
    -------
    SparseMatrix
        Sparse matrix of shape ``(L, N)``
    """
    assert (
        A.shape[1] == D.shape[0]
    ), f"The second dimension of SparseMatrix should be equal to the first \
    dimension of DiagMatrix in matmul(SparseMatrix, DiagMatrix), but the \
    shapes of SparseMatrix and DiagMatrix are {A.shape} and {D.shape} \
    respectively."
    assert (
        D.shape[0] == D.shape[1]
    ), f"The DiagMatrix should be a square in matmul(SparseMatrix, DiagMatrix) \
    but got {D.shape}."
    return val_like(A, D.val[A.col] * A.val)


def _diag_sparse_mm(D, A):
    """Internal function for multiplying a diagonal matrix by a sparse matrix.

    Parameters
    ----------
    D : DiagMatrix
        Diagonal matrix of shape ``(L, M)``
    A : SparseMatrix
        Sparse matrix of shape ``(M, N)``

    Returns
    -------
    SparseMatrix
        Sparse matrix of shape ``(L, N)``
    """
    assert (
        D.shape[1] == A.shape[0]
    ), f"The second dimension of DiagMatrix should be equal to the first \
    dimension of SparseMatrix in matmul(DiagMatrix, SparseMatrix), but the \
    shapes of DiagMatrix and SparseMatrix are {D.shape} and {A.shape} \
    respectively."
    assert (
        D.shape[0] == D.shape[1]
    ), f"The DiagMatrix should be a square in matmul(DiagMatrix, SparseMatrix) \
    but got {D.shape}."
    return val_like(A, D.val[A.row] * A.val)


def spspmm(
    A: Union[SparseMatrix, DiagMatrix], B: Union[SparseMatrix, DiagMatrix]
) -> Union[SparseMatrix, DiagMatrix]:
    """Multiplies a sparse matrix by a sparse matrix, equivalent to ``A @ B``.

    The non-zero values of the two sparse matrices must be 1D.

    Parameters
    ----------
    A : SparseMatrix or DiagMatrix
        Sparse matrix of shape ``(L, M)``
    B : SparseMatrix or DiagMatrix
        Sparse matrix of shape ``(M, N)``

    Returns
    -------
    SparseMatrix or DiagMatrix
        Matrix of shape ``(L, N)``. It is a DiagMatrix object if both matrices
        are DiagMatrix objects, otherwise a SparseMatrix object.

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
        A, (SparseMatrix, DiagMatrix)
    ), f"Expect A1 to be a SparseMatrix or DiagMatrix object, got {type(A)}."
    assert isinstance(
        B, (SparseMatrix, DiagMatrix)
    ), f"Expect A2 to be a SparseMatrix or DiagMatrix object, got {type(B)}."

    if isinstance(A, DiagMatrix) and isinstance(B, DiagMatrix):
        return _diag_diag_mm(A, B)
    if isinstance(A, DiagMatrix):
        return _diag_sparse_mm(A, B)
    if isinstance(B, DiagMatrix):
        return _sparse_diag_mm(A, B)
    return SparseMatrix(
        torch.ops.dgl_sparse.spspmm(A.c_sparse_matrix, B.c_sparse_matrix)
    )


def matmul(
    A: Union[torch.Tensor, SparseMatrix, DiagMatrix],
    B: Union[torch.Tensor, SparseMatrix, DiagMatrix],
) -> Union[torch.Tensor, SparseMatrix, DiagMatrix]:
    """Multiplies two dense/sparse/diagonal matrices, equivalent to ``A @ B``.

    The supported combinations are shown as follows.

    +--------------+--------+------------+--------------+
    |   A \\ B   | Tensor | DiagMatrix | SparseMatrix |
    +--------------+--------+------------+--------------+
    |    Tensor    |   ✅   |     🚫     |      🚫      |
    +--------------+--------+------------+--------------+
    | SparseMatrix |   ✅   |     ✅     |      ✅      |
    +--------------+--------+------------+--------------+
    |  DiagMatrix  |   ✅   |     ✅     |      ✅      |
    +--------------+--------+------------+--------------+

    * If both matrices are torch.Tensor, it calls \
        :func:`torch.matmul()`. The result is a dense matrix.

    * If both matrices are sparse or diagonal, it calls \
        :func:`dgl.sparse.spspmm`. The result is a sparse matrix.

    * If :attr:`A` is sparse or diagonal while :attr:`B` is dense, it \
        calls :func:`dgl.sparse.spmm`. The result is a dense matrix.

    * The operator supports batched sparse-dense matrix multiplication. In \
        this case, the sparse or diagonal matrix :attr:`A` should have shape \
        ``(L, M)``, where the non-zero values have a batch dimension ``K``. \
        The dense matrix :attr:`B` should have shape ``(M, N, K)``. The output \
        is a dense matrix of shape ``(L, N, K)``.

    * Sparse-sparse matrix multiplication does not support batched computation.

    Parameters
    ----------
    A : torch.Tensor, SparseMatrix or DiagMatrix
        The first matrix.
    B : torch.Tensor, SparseMatrix, or DiagMatrix
        The second matrix.

    Returns
    -------
    torch.Tensor, SparseMatrix or DiagMatrix
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
    >>> val = torch.randn(len(row))
    >>> A = dglsp.spmatrix(indices, val)
    >>> X = torch.randn(2, 3)
    >>> result = dglsp.matmul(A, X)
    >>> type(result)
    <class 'torch.Tensor'>
    >>> result.shape
    torch.Size([2, 3])

    Multiplies a sparse matrix with a sparse matrix.

    >>> indices1 = torch.tensor([[0, 1, 1], [1, 0, 1]])
    >>> val1 = torch.ones(len(row1))
    >>> A = dglsp.spmatrix(indices1, val1)
    >>> indices2 = torch.tensor([[0, 1, 1], [0, 2, 1]])
    >>> val2 = torch.ones(len(row2))
    >>> B = dglsp.spmatrix(indices2, val2)
    >>> result = dglsp.matmul(A, B)
    >>> type(result)
    <class 'dgl.sparse.sparse_matrix.SparseMatrix'>
    >>> result.shape
    (2, 3)
    """
    assert isinstance(A, (torch.Tensor, SparseMatrix, DiagMatrix)), (
        f"Expect arg1 to be a torch.Tensor, SparseMatrix, or DiagMatrix object,"
        f"got {type(A)}."
    )
    assert isinstance(B, (torch.Tensor, SparseMatrix, DiagMatrix)), (
        f"Expect arg2 to be a torch Tensor, SparseMatrix, or DiagMatrix"
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
    if isinstance(A, DiagMatrix) and isinstance(B, DiagMatrix):
        return _diag_diag_mm(A, B)
    return spspmm(A, B)


SparseMatrix.__matmul__ = matmul
DiagMatrix.__matmul__ = matmul
