"""Unary ops for SparseMatrix"""
# pylint: disable=invalid-name
import numpy as np
import torch

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import inv as scipy_inv

from .sp_matrix import SparseMatrix, create_from_coo
from ..convert import graph
from ..ops.edge_softmax import edge_softmax

def neg(A: SparseMatrix) -> SparseMatrix:
    """Return a new sparse matrix with negative elements.

    Returns
    -------
    SparseMatrix
        Negative of the sparse matrix.

    Examples
    --------

    >>> row = torch.tensor([1, 1, 3])
    >>> col = torch.tensor([1, 2, 3])
    >>> val = torch.tensor([1., 1., 2.])
    >>> A = create_from_coo(row, col, val)
    >>> A = -A
    >>> print(A)
    SparseMatrix(indices=tensor([[1, 1, 3],
                                 [1, 2, 3]]),
                 values=tensor([-1., -1., -2.]),
                 shape=(4, 4), nnz=3)
    """
    return create_from_coo(row=A.row,
                           col=A.col,
                           val=-A.val,
                           shape=A.shape)

def inv(A: SparseMatrix) -> SparseMatrix:
    """Compute the inverse.

    Only non-singular square matrices with values of shape (nnz) are supported.

    Returns
    -------
    SparseMatrix
        Inverse of the sparse matrix.

    Examples
    --------

    [[1, 0],
     [1, 2]]

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 1])
    >>> val = torch.tensor([1, 1, 2])
    >>> A = create_from_coo(row, col, val)

    [[1,    0  ],
     [-0.5, 0.5]]

    >>> A_inv = A.inv()
    >>> print(A_inv)
    SparseMatrix(indices=tensor([[0, 1, 1],
                                 [0, 0, 1]]),
                 values=tensor([1.0000, -0.5000, 0.5000]),
                 shape=(2, 2), nnz=3)
    """
    num_rows, num_cols = A.shape
    assert num_rows == num_cols, 'Expect a square matrix, got shape {}'.format(A.shape)
    assert len(A.val.shape) == 1, 'inv only supports matrices with 1D val'

    val = A.val.cpu().numpy()
    row = A.row.cpu().numpy()
    col = A.col.cpu().numpy()
    # The computation is more efficient with CSC format.
    mat = coo_matrix((val, (row, col)), dtype=val.dtype).tocsc()
    mat_inv = scipy_inv(mat)
    row, col = mat_inv.nonzero()
    val = mat_inv[row, col]
    val = np.asarray(val).squeeze(0)
    dev = A.device

    return create_from_coo(row=torch.from_numpy(row).to(dev),
                           col=torch.from_numpy(col).to(dev),
                           val=torch.from_numpy(val).to(dev),
                           shape=A.shape)

def softmax(A: SparseMatrix) -> SparseMatrix:
    """Apply row-wise softmax to the nonzero entries of the sparse matrix.

    If :attr:`A.val` takes shape :attr:`(nnz, D)`, then the output matrix
    :attr:`A'` and :attr:`A'.val` take the same shape as :attr:`A` and :attr:`A.val`.
    :attr:`A'.val[:, i]` is calculated based on :attr:`A.val[:, i]`.

    Parameters
    ----------
    A : SparseMatrix
        The input sparse matrix

    Returns
    -------
    SparseMatrix
        The result, whose shape is the same as :attr:`A`

    Examples
    --------

    Case1: matrix with values of shape (nnz)

    >>> row = torch.tensor([0, 0, 1, 2])
    >>> col = torch.tensor([1, 2, 2, 0])
    >>> val = torch.ones(len(row))
    >>> A = create_from_coo(row, col, val)
    >>> result = A.softmax()
    >>> result.val
    tensor([0.5000, 0.5000, 1.0000, 1.0000])
    >>> result.shape
    (3, 3)

    Case2: matrix with values of shape (nnz, D)

    >>> row = torch.tensor([0, 0, 1, 2])
    >>> col = torch.tensor([1, 2, 2, 0])
    >>> val = torch.ones(len(row), 2)
    >>> A = create_from_coo(row, col, val)
    >>> result = A.softmax()
    >>> result.val
    tensor([[0.5000, 0.5000],
            [0.5000, 0.5000],
            [1.0000, 1.0000],
            [1.0000, 1.0000]])
    >>> result.shape
    (3, 3)
    """
    g = graph((A.col, A.row))
    return create_from_coo(A.row,
                           A.col,
                           edge_softmax(g, A.val),
                           A.shape)

SparseMatrix.__neg__ = neg
SparseMatrix.inv = inv
SparseMatrix.softmax = softmax
