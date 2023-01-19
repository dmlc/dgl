"""DGL unary operators for diagonal matrix module."""
# pylint: disable= invalid-name
from .diag_matrix import diag, DiagMatrix


def neg(D: DiagMatrix) -> DiagMatrix:
    """Returns a new diagonal matrix with the negation of the original nonzero
    values, equivalent to ``-D``.

    Returns
    -------
    DiagMatrix
        Negation of the diagonal matrix

    Examples
    --------

    >>> val = torch.arange(3).float()
    >>> D = dglsp.diag(val)
    >>> D = -D
    DiagMatrix(val=tensor([-0., -1., -2.]),
               shape=(3, 3))
    """
    return diag(-D.val, D.shape)


def inv(D: DiagMatrix) -> DiagMatrix:
    """Returns the inverse of the diagonal matrix.

    This function only supports square matrices with scalar nonzero values.

    Returns
    -------
    DiagMatrix
        Inverse of the diagonal matrix

    Examples
    --------

    >>> val = torch.arange(1, 4).float()
    >>> D = dglsp.diag(val)
    >>> D = D.inv()
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
