"""Unary ops for DiagMatrix"""
# pylint: disable=invalid-name
from .diag_matrix import DiagMatrix

def neg(D):
    """Return a new diagonal matrix with negative elements.

    Returns
    -------
    DiagMatrix
        Negative of the diagonal matrix.

    Examples
    --------

    >>> val = torch.arange(3).float()
    >>> mat = diag(val)
    >>> mat = -mat
    >>> print(mat)
    DiagMatrix(val=tensor([-0., -1., -2.]),
               shape=(3, 3))
    """
    return DiagMatrix(-D.val, D.shape)

def inv(D):
    """Compute the inverse.

    Only square matrices with values of shape (nnz) are supported.

    Returns
    -------
    DiagMatrix
        Inverse of the diagonal matrix.

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
    assert num_rows == num_cols, f'Expect a square matrix, got shape {D.shape}'
    assert len(D.val.shape) == 1, 'inv only supports matrices with 1D val'

    return DiagMatrix(1. / D.val, D.shape)

DiagMatrix.neg = neg
DiagMatrix.__neg__ = neg
DiagMatrix.inv = inv
