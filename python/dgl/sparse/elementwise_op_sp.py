"""DGL elementwise operators for sparse matrix module."""
import torch

from .sparse_matrix import SparseMatrix, val_like
from .utils import is_scalar, Scalar


def spsp_add(A, B):
    """Invoke C++ sparse library for addition"""
    return SparseMatrix(
        torch.ops.dgl_sparse.spsp_add(A.c_sparse_matrix, B.c_sparse_matrix)
    )


def sp_add(A: SparseMatrix, B: SparseMatrix) -> SparseMatrix:
    """Elementwise addition

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    B : SparseMatrix
        Sparse matrix

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------

    >>> indices = torch.tensor([[1, 0, 2], [0, 3, 2]])
    >>> val = torch.tensor([10, 20, 30])
    >>> A = dglsp.spmatrix(indices, val, shape=(3, 4))
    >>> A + A
    SparseMatrix(indices=tensor([[0, 1, 2],
                                 [3, 0, 2]]),
                 values=tensor([40, 20, 60]),
                 shape=(3, 4), nnz=3)
    """
    # Python falls back to B.__radd__ then TypeError when NotImplemented is
    # returned.
    return spsp_add(A, B) if isinstance(B, SparseMatrix) else NotImplemented


def sp_sub(A: SparseMatrix, B: SparseMatrix) -> SparseMatrix:
    """Elementwise subtraction

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    B : SparseMatrix
        Sparse matrix

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------

    >>> indices = torch.tensor([[1, 0, 2], [0, 3, 2]])
    >>> val = torch.tensor([10, 20, 30])
    >>> val2 = torch.tensor([5, 10, 15])
    >>> A = dglsp.spmatrix(indices, val, shape=(3, 4))
    >>> B = dglsp.spmatrix(indices, val2, shape=(3, 4))
    >>> A - B
    SparseMatrix(indices=tensor([[0, 1, 2],
                                 [3, 0, 2]]),
                 values=tensor([10, 5, 15]),
                 shape=(3, 4), nnz=3)
    """
    # Python falls back to B.__rsub__ then TypeError when NotImplemented is
    # returned.
    return spsp_add(A, -B) if isinstance(B, SparseMatrix) else NotImplemented


def sp_mul(A: SparseMatrix, B: Scalar) -> SparseMatrix:

    """Elementwise multiplication

    Parameters
    ----------
    A : SparseMatrix
        First operand
    B : Scalar
        Second operand

    Returns
    -------
    SparseMatrix
        Result of A * B

    Examples
    --------

    >>> indices = torch.tensor([[1, 0, 2], [0, 3, 2]])
    >>> val = torch.tensor([1, 2, 3])
    >>> A = dglsp.spmatrix(indices, val, shape=(3, 4))

    >>> A * 2
    SparseMatrix(indices=tensor([[1, 0, 2],
                                 [0, 3, 2]]),
    values=tensor([2, 4, 6]),
    shape=(3, 4), nnz=3)

    >>> 2 * A
    SparseMatrix(indices=tensor([[1, 0, 2],
                                 [0, 3, 2]]),
    values=tensor([2, 4, 6]),
    shape=(3, 4), nnz=3)
    """
    if is_scalar(B):
        return val_like(A, A.val * B)
    # Python falls back to B.__rmul__(A) then TypeError when NotImplemented is
    # returned.
    # So this also handles the case of scalar * SparseMatrix since we set
    # SparseMatrix.__rmul__ to be the same as SparseMatrix.__mul__.
    return NotImplemented


def sp_div(A: SparseMatrix, B: Scalar) -> SparseMatrix:
    """Elementwise division

    Parameters
    ----------
    A : SparseMatrix
        First operand
    B : Scalar
        Second operand

    Returns
    -------
    SparseMatrix
        Result of A / B

    Examples
    --------
    >>> indices = torch.tensor([[1, 0, 2], [0, 3, 2]])
    >>> val = torch.tensor([1, 2, 3])
    >>> A = dglsp.spmatrix(indices, val, shape=(3, 4))
    >>> A / 2
    SparseMatrix(indices=tensor([[1, 0, 2],
                                 [0, 3, 2]]),
                 values=tensor([0.5000, 1.0000, 1.5000]),
                 shape=(3, 4), nnz=3)
    """
    if is_scalar(B):
        return val_like(A, A.val / B)
    # Python falls back to B.__rtruediv__(A) then TypeError when NotImplemented
    # is returned.
    return NotImplemented


def sp_power(A: SparseMatrix, scalar: Scalar) -> SparseMatrix:
    """Take the power of each nonzero element and return a sparse matrix with
    the result.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    scalar : float or int
        Exponent

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    >>> indices = torch.tensor([[1, 0, 2], [0, 3, 2]])
    >>> val = torch.tensor([10, 20, 30])
    >>> A = dglsp.spmatrix(indices, val)
    >>> A ** 2
    SparseMatrix(indices=tensor([[1, 0, 2],
                                 [0, 3, 2]]),
    values=tensor([100, 400, 900]),
    shape=(3, 4), nnz=3)
    """
    # Python falls back to scalar.__rpow__ then TypeError when NotImplemented
    # is returned.
    return val_like(A, A.val**scalar) if is_scalar(scalar) else NotImplemented


SparseMatrix.__add__ = sp_add
SparseMatrix.__sub__ = sp_sub
SparseMatrix.__mul__ = sp_mul
SparseMatrix.__rmul__ = sp_mul
SparseMatrix.__truediv__ = sp_div
SparseMatrix.__pow__ = sp_power
