"""DGL unary operators for sparse matrix module."""
from .sparse_matrix import SparseMatrix, val_like


def neg(A: SparseMatrix) -> SparseMatrix:
    """Return a new sparse matrix with the negation of the original nonzero
    values.

    Returns
    -------
    SparseMatrix
        Negation of the sparse matrix

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
    return val_like(A, -A.val)


SparseMatrix.neg = neg
SparseMatrix.__neg__ = neg
