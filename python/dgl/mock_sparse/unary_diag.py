"""Unary ops for DiagMatrix"""
# pylint: disable=invalid-name
import torch

from .diag_matrix import DiagMatrix, diag

def neg(D: DiagMatrix) -> DiagMatrix:
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
    return diag(-D.val, D.shape)

def inv(D: DiagMatrix) -> DiagMatrix:
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

    return diag(1. / D.val, D.shape)

def softmax(D: DiagMatrix) -> DiagMatrix:
    """Apply row-wise softmax to the nonzero entries of the diagonal matrix.

    The result will be a diagonal matrix with one-valued diagonal.

    Parameters
    ----------
    D : DiagMatrix
        The input diagonal matrix

    Returns
    -------
    DiagMatrix
        The result.

    Examples
    --------

    Case1: matrix with values of shape (nnz)

    >>> val = torch.randn(3)
    >>> D = diag(val)
    >>> result = D.softmax()
    >>> result.val
    tensor([1., 1., 1.])
    >>> result.shape
    (3, 3)

    Case2: matrix with values of shape (nnz, D)

    >>> val = torch.randn(3, 4)
    >>> D = diag(val)
    >>> result = D.softmax()
    >>> result.val
    tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])
    >>> result.shape
    (3, 3)
    """
    return diag(torch.ones_like(D.val), D.shape)

DiagMatrix.__neg__ = neg
DiagMatrix.inv = inv
DiagMatrix.softmax = softmax
