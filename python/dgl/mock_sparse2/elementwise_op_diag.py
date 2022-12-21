"""DGL elementwise operators for diagonal matrix module."""
from typing import Union

from .diag_matrix import DiagMatrix, diag
from .sparse_matrix import SparseMatrix

__all__ = ["diag_add", "diag_sub", "diag_mul", "diag_div", "diag_power"]


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
    >>> D1 = diag(torch.arange(1, 4))
    >>> D2 = diag(torch.arange(10, 13))
    >>> D1 + D2
    DiagMatrix(val=tensor([11, 13, 15]),
    shape=(3, 3))
    """
    if isinstance(D1, DiagMatrix) and isinstance(D2, DiagMatrix):
        assert D1.shape == D2.shape, (
            "The shape of diagonal matrix D1 "
            f"{D1.shape} and D2 {D2.shape} must match."
        )
        return diag(D1.val + D2.val, D1.shape)
    elif isinstance(D1, DiagMatrix) and isinstance(D2, SparseMatrix):
        assert D1.shape == D2.shape, (
            "The shape of diagonal matrix D1 "
            f"{D1.shape} and sparse matrix D2 {D2.shape} must match."
        )
        D1 = D1.as_sparse()
        return D1 + D2
    raise RuntimeError(
        "Elementwise addition between "
        f"{type(D1)} and {type(D2)} is not supported."
    )


def diag_sub(D1: DiagMatrix, D2: DiagMatrix) -> DiagMatrix:
    """Elementwise subtraction

    Parameters
    ----------
    D1 : DiagMatrix
        Diagonal matrix
    D2 : DiagMatrix
        Diagonal matrix

    Returns
    -------
    DiagMatrix
        Diagonal matrix

    Examples
    --------
    >>> D1 = diag(torch.arange(1, 4))
    >>> D2 = diag(torch.arange(10, 13))
    >>> D1 - D2
    DiagMatrix(val=tensor([-9, -9, -9]),
    shape=(3, 3))
    """
    if isinstance(D1, DiagMatrix) and isinstance(D2, DiagMatrix):
        assert D1.shape == D2.shape, (
            "The shape of diagonal matrix D1 "
            f"{D1.shape} and D2 {D2.shape} must match."
        )
        return diag(D1.val - D2.val, D1.shape)
    raise RuntimeError(
        "Elementwise subtraction between "
        f"{type(D1)} and {type(D2)} is not supported."
    )


def diag_mul(
    D1: Union[DiagMatrix, float, int], D2: Union[DiagMatrix, float, int]
) -> DiagMatrix:
    """Elementwise multiplication

    Parameters
    ----------
    D1 : DiagMatrix or float or int
        Diagonal matrix or scalar value
    D2 : DiagMatrix or float or int
        Diagonal matrix or scalar value

    Returns
    -------
    DiagMatrix
        Diagonal matrix

    Examples
    --------
    >>> D = diag(torch.arange(1, 4))
    >>> D * 2.5
    DiagMatrix(val=tensor([2.5000, 5.0000, 7.5000]),
    shape=(3, 3))
    >>> 2 * D
    DiagMatrix(val=tensor([2, 4, 6]),
    shape=(3, 3))
    """
    if isinstance(D1, DiagMatrix) and isinstance(D2, DiagMatrix):
        assert D1.shape == D2.shape, (
            "The shape of diagonal matrix D1 "
            f"{D1.shape} and D2 {D2.shape} must match."
        )
        return diag(D1.val * D2.val, D1.shape)
    elif isinstance(D1, DiagMatrix) and isinstance(D2, (float, int)):
        return diag(D1.val * D2, D1.shape)
    elif isinstance(D1, (float, int)) and isinstance(D2, DiagMatrix):
        return diag(D1 * D2.val, D2.shape)

    raise RuntimeError(
        "Elementwise multiplication between "
        f"{type(D1)} and {type(D2)} is not supported."
    )


def diag_div(D1: DiagMatrix, D2: Union[DiagMatrix, float, int]) -> DiagMatrix:
    """Elementwise division of a diagonal matrix by a diagonal matrix or a
    scalar

    Parameters
    ----------
    D1 : DiagMatrix
        Diagonal matrix
    D2 : DiagMatrix or float or int
        Diagonal matrix or scalar value. If :attr:`D2` is a DiagMatrix,
        division is only applied to the diagonal elements.

    Returns
    -------
    DiagMatrix
        diagonal matrix

    Examples
    --------
    >>> D1 = diag(torch.arange(1, 4))
    >>> D2 = diag(torch.arange(10, 13))
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
    elif isinstance(D2, (float, int)):
        assert D2 != 0, "Division by zero is not allowed."
        return diag(D1.val / D2, D1.shape)

    raise RuntimeError(
        f"Elementwise division between a diagonal matrix and {type(D2)} is "
        "not supported."
    )


def diag_rdiv(D1: DiagMatrix, D2: Union[float, int]):
    """Function for preventing elementwise division of a scalar by a diagonal
    matrix

    Parameters
    ----------
    D1 : DiagMatrix
        Diagonal matrix
    D2 : float or int
        Scalar value
    """
    raise RuntimeError(
        f"Elementwise division of {type(D2)} by a diagonal matrix is not "
        "supported."
    )


# pylint: disable=invalid-name
def diag_power(D: DiagMatrix, scalar: Union[float, int]) -> DiagMatrix:
    """Take the power of each nonzero element and return a diagonal matrix with
    the result.

    Parameters
    ----------
    D : DiagMatrix
        Diagonal matrix
    scalar : float or int
        Exponent

    Returns
    -------
    DiagMatrix
        Diagonal matrix

    Examples
    --------
    >>> D = diag(torch.arange(1, 4))
    >>> D ** 2
    DiagMatrix(val=tensor([1, 4, 9]),
    shape=(3, 3))
    """
    if isinstance(scalar, (float, int)):
        return diag(D.val**scalar, D.shape)

    raise RuntimeError(
        f"Raising a diagonal matrix to exponent {type(scalar)} is not allowed."
    )


def diag_rpower(D: DiagMatrix, scalar: Union[float, int]):
    """Function for preventing raising a scalar to a diagonal matrix exponent

    Parameters
    ----------
    D : DiagMatrix
        Diagonal matrix
    scalar : float or int
        Scalar
    """
    raise RuntimeError(
        f"Raising {type(scalar)} to a diagonal matrix component is not "
        "allowed."
    )


DiagMatrix.__add__ = diag_add
DiagMatrix.__radd__ = diag_add
DiagMatrix.__sub__ = diag_sub
DiagMatrix.__rsub__ = diag_sub
DiagMatrix.__mul__ = diag_mul
DiagMatrix.__rmul__ = diag_mul
DiagMatrix.__truediv__ = diag_div
DiagMatrix.__rtruediv__ = diag_rdiv
DiagMatrix.__pow__ = diag_power
DiagMatrix.__rpow__ = diag_rpower
