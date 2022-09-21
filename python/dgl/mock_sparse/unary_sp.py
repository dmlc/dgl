import numpy as np
import torch

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import inv as scipy_inv

from .sp_matrix import SparseMatrix

def neg(A):
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
    return SparseMatrix(A._row, A._col, -A._val, A.shape)

def inv(A):
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
    assert len(A._val.shape) == 1, 'inv only supports matrices with 1D val'

    val = A._val.numpy()
    row = A._row.numpy()
    col = A._col.numpy()
    # The computation is more efficient with CSC format.
    mat = coo_matrix((val, (row, col)), dtype=val.dtype).tocsc()
    mat_inv = scipy_inv(mat)
    row, col = mat_inv.nonzero()
    val = mat_inv[row, col]
    val = np.asarray(val).squeeze(0)
    dev = A.device
    return SparseMatrix(torch.from_numpy(row).to(dev),
                        torch.from_numpy(col).to(dev),
                        torch.from_numpy(val).to(dev),
                        A.shape)

SparseMatrix.neg = neg
SparseMatrix.__neg__ = neg
SparseMatrix.inv = inv
