"""DGL unary operators for diagonal matrix module."""
# pylint: disable= invalid-name
from .diag_matrix import DiagMatrix, diag


def neg(D: DiagMatrix) -> DiagMatrix:
    """Return a new diagonal matrix with the negation of the original nonzero
    values.

    Returns
    -------
    DiagMatrix
        Negation of the diagonal matrix

    Examples
    --------

    >>> val = torch.arange(3).float()
    >>> mat = diag(val)
    >>> mat = -mat
    >>> print(mat)
    DiagMatrix(val=tensor([-0., -1., -2.]),
    shape=(3, 3))
    """
    return diag(-D.val, D.shape)


def inv(D: DiagMatrix) -> DiagMatrix:
    """Return the inverse of the diagonal matrix.

    This function only supports square matrices with scalar nonzero values.

    Returns
    -------
    DiagMatrix
        Inverse of the diagonal matrix

    Examples
    --------

    >>> val = torch.arange(1, 4).float()
    >>> mat = diag(val)
    >>> mat = mat.inv()
    >>> print(mat)
    DiagMatrix(val=tensor([1.0000, 0.5000, 0.3333]),
    shape=(3, 3))
    """
    num_rows, num_cols = D.shape
    assert num_rows == num_cols, f"Expect a square matrix, got shape {D.shape}"
    assert len(D.val.shape) == 1, "inv only supports 1D nonzero val"

    return diag(1.0 / D.val, D.shape)


DiagMatrix.neg = neg
DiagMatrix.__neg__ = neg
DiagMatrix.inv = inv
