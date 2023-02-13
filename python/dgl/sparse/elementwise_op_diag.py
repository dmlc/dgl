"""DGL elementwise operators for diagonal matrix module."""
from typing import Union

from .diag_matrix import diag, DiagMatrix
from .sparse_matrix import SparseMatrix
from .utils import is_scalar, Scalar


def diag_add(
    D1: DiagMatrix, D2: Union[DiagMatrix, SparseMatrix]
) -> Union[DiagMatrix, SparseMatrix]:
    """Elementwise addition

    Parameters
    ----------
    D1 : DiagMatrix
        Diagonal matrix
    D2 : DiagMatrix or SparseMatrix
        Diagonal matrix or sparse matrix

    Returns
    -------
    DiagMatrix or SparseMatrix
        Diagonal matrix or sparse matrix, same as D2

    Examples
    --------
    >>> D1 = dglsp.diag(torch.arange(1, 4))
    >>> D2 = dglsp.diag(torch.arange(10, 13))
    >>> D1 + D2
    DiagMatrix(val=tensor([11, 13, 15]),
               shape=(3, 3))
    """
    if isinstance(D2, DiagMatrix):
        assert D1.shape == D2.shape, (
            "The shape of diagonal matrix D1 "
            f"{D1.shape} and D2 {D2.shape} must match."
        )
        return diag(D1.val + D2.val, D1.shape)
    elif isinstance(D2, SparseMatrix):
        assert D1.shape == D2.shape, (
            "The shape of diagonal matrix D1 "
            f"{D1.shape} and sparse matrix D2 {D2.shape} must match."
        )
        D1 = D1.to_sparse()
        return D1 + D2
    # Python falls back to D2.__radd__(D1) then TypeError when NotImplemented
    # is returned.
    return NotImplemented


def diag_sub(
    D1: DiagMatrix, D2: Union[DiagMatrix, SparseMatrix]
) -> Union[DiagMatrix, SparseMatrix]:
    """Elementwise subtraction

    Parameters
    ----------
    D1 : DiagMatrix
        Diagonal matrix
    D2 : DiagMatrix or SparseMatrix
        Diagonal matrix or sparse matrix

    Returns
    -------
    DiagMatrix or SparseMatrix
        Diagonal matrix or sparse matrix, same as D2

    Examples
    --------
    >>> D1 = dglsp.diag(torch.arange(1, 4))
    >>> D2 = dglsp.diag(torch.arange(10, 13))
    >>> D1 - D2
    DiagMatrix(val=tensor([-9, -9, -9]),
               shape=(3, 3))
    """
    if isinstance(D2, DiagMatrix):
        assert D1.shape == D2.shape, (
            "The shape of diagonal matrix D1 "
            f"{D1.shape} and D2 {D2.shape} must match."
        )
        return diag(D1.val - D2.val, D1.shape)
    elif isinstance(D2, SparseMatrix):
        assert D1.shape == D2.shape, (
            "The shape of diagonal matrix D1 "
            f"{D1.shape} and sparse matrix D2 {D2.shape} must match."
        )
        D1 = D1.to_sparse()
        return D1 - D2
    # Python falls back to D2.__rsub__(D1) then TypeError when NotImplemented
    # is returned.
    return NotImplemented


def diag_rsub(
    D1: DiagMatrix, D2: Union[DiagMatrix, SparseMatrix]
) -> Union[DiagMatrix, SparseMatrix]:
    """Elementwise subtraction in the opposite direction (``D2 - D1``)

    Parameters
    ----------
    D1 : DiagMatrix
        Diagonal matrix
    D2 : DiagMatrix or SparseMatrix
        Diagonal matrix or sparse matrix

    Returns
    -------
    DiagMatrix or SparseMatrix
        Diagonal matrix or sparse matrix, same as D2

    Examples
    --------
    >>> D1 = dglsp.diag(torch.arange(1, 4))
    >>> D2 = dglsp.diag(torch.arange(10, 13))
    >>> D2 - D1
    DiagMatrix(val=tensor([-9, -9, -9]),
               shape=(3, 3))
    """
    return -(D1 - D2)


def diag_mul(D1: DiagMatrix, D2: Union[DiagMatrix, Scalar]) -> DiagMatrix:
    """Elementwise multiplication

    Parameters
    ----------
    D1 : DiagMatrix
        Diagonal matrix
    D2 : DiagMatrix or Scalar
        Diagonal matrix or scalar value

    Returns
    -------
    DiagMatrix
        Diagonal matrix

    Examples
    --------
    >>> D = dglsp.diag(torch.arange(1, 4))
    >>> D * 2.5
    DiagMatrix(val=tensor([2.5000, 5.0000, 7.5000]),
               shape=(3, 3))
    >>> 2 * D
    DiagMatrix(val=tensor([2, 4, 6]),
               shape=(3, 3))
    """
    if isinstance(D2, DiagMatrix):
        assert D1.shape == D2.shape, (
            "The shape of diagonal matrix D1 "
            f"{D1.shape} and D2 {D2.shape} must match."
        )
        return diag(D1.val * D2.val, D1.shape)
    elif is_scalar(D2):
        return diag(D1.val * D2, D1.shape)
    else:
        # Python falls back to D2.__rmul__(D1) then TypeError when
        # NotImplemented is returned.
        return NotImplemented


def diag_div(D1: DiagMatrix, D2: Union[DiagMatrix, Scalar]) -> DiagMatrix:
    """Elementwise division of a diagonal matrix by a diagonal matrix or a
    scalar

    Parameters
    ----------
    D1 : DiagMatrix
        Diagonal matrix
    D2 : DiagMatrix or Scalar
        Diagonal matrix or scalar value. If :attr:`D2` is a DiagMatrix,
        division is only applied to the diagonal elements.

    Returns
    -------
    DiagMatrix
        diagonal matrix

    Examples
    --------
    >>> D1 = dglsp.diag(torch.arange(1, 4))
    >>> D2 = dglsp.diag(torch.arange(10, 13))
    >>> D1 / D2
    DiagMatrix(val=tensor([0.1000, 0.1818, 0.2500]),
               shape=(3, 3))
    >>> D1 / 2.5
    DiagMatrix(val=tensor([0.4000, 0.8000, 1.2000]),
               shape=(3, 3))
    """
    if isinstance(D2, DiagMatrix):
        assert D1.shape == D2.shape, (
            f"The shape of diagonal matrix D1 {D1.shape} and D2 {D2.shape} "
            "must match."
        )
        return diag(D1.val / D2.val, D1.shape)
    elif is_scalar(D2):
        assert D2 != 0, "Division by zero is not allowed."
        return diag(D1.val / D2, D1.shape)
    else:
        # Python falls back to D2.__rtruediv__(D1) then TypeError when
        # NotImplemented is returned.
        return NotImplemented


# pylint: disable=invalid-name
def diag_power(D: DiagMatrix, scalar: Scalar) -> DiagMatrix:
    """Take the power of each nonzero element and return a diagonal matrix with
    the result.

    Parameters
    ----------
    D : DiagMatrix
        Diagonal matrix
    scalar : Scalar
        Exponent

    Returns
    -------
    DiagMatrix
        Diagonal matrix

    Examples
    --------
    >>> D = dglsp.diag(torch.arange(1, 4))
    >>> D ** 2
    DiagMatrix(val=tensor([1, 4, 9]),
               shape=(3, 3))
    """
    return (
        diag(D.val**scalar, D.shape) if is_scalar(scalar) else NotImplemented
    )


DiagMatrix.__add__ = diag_add
DiagMatrix.__radd__ = diag_add
DiagMatrix.__sub__ = diag_sub
DiagMatrix.__rsub__ = diag_rsub
DiagMatrix.__mul__ = diag_mul
DiagMatrix.__rmul__ = diag_mul
DiagMatrix.__truediv__ = diag_div
DiagMatrix.__pow__ = diag_power
