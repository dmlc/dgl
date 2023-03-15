"""DGL unary operators for sparse matrix module."""
from .sparse_matrix import diag, SparseMatrix, val_like


def neg(A: SparseMatrix) -> SparseMatrix:
    """Returns a new sparse matrix with the negation of the original nonzero
    values, equivalent to ``-A``.

    Returns
    -------
    SparseMatrix
        Negation of the sparse matrix

    Examples
    --------

    >>> indices = torch.tensor([[1, 1, 3], [1, 2, 3]])
    >>> val = torch.tensor([1., 1., 2.])
    >>> A = dglsp.spmatrix(indices, val)
    >>> A = -A
    SparseMatrix(indices=tensor([[1, 1, 3],
                                 [1, 2, 3]]),
                 values=tensor([-1., -1., -2.]),
                 shape=(4, 4), nnz=3)
    """
    return val_like(A, -A.val)


def inv(A: SparseMatrix) -> SparseMatrix:
    """Returns the inverse of the sparse matrix.

    This function only supports square diagonal matrices with scalar nonzero
    values.

    Returns
    -------
    SparseMatrix
        Inverse of the sparse matrix

    Examples
    --------

    >>> val = torch.arange(1, 4).float()
    >>> D = dglsp.diag(val)
    >>> D.inv()
    SparseMatrix(indices=tensor([[0, 1, 2],
                                 [0, 1, 2]]),
                 values=tensor([1., 2., 3.]),
                 shape=(3, 3), nnz=3)
    """
    num_rows, num_cols = A.shape
    assert A.is_diag(), "Non-diagonal sparse matrix does not support inversion."
    assert num_rows == num_cols, f"Expect a square matrix, got shape {A.shape}"
    assert len(A.val.shape) == 1, "inv only supports 1D nonzero val"

    return diag(1.0 / A.val, A.shape)


SparseMatrix.neg = neg
SparseMatrix.__neg__ = neg
SparseMatrix.inv = inv
