"""dgl elementwise operators for sparse matrix module."""
import torch
from .sp_matrix import *

__all__ = ['add', 'sub', 'mul', 'div', 'rdiv', 'power', 'rpower']

def add(A, B):
    """Elementwise addition.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    B : SparseMatrix
        Sparse matrix

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    Case 1: Add two matrices of same sparsity structure

    >>> rowA = torch.tensor([1, 0, 2, 7, 1])
    >>> colA = torch.tensor([0, 49, 2, 1, 7])
    >>> valA = torch.tensor([10, 20, 30, 40, 50])
    >>> A = SparseMatrix(rowA, colA, valA, shape=(10, 50))
    >>> A + A
    SparseMatrix(indices=tensor([[ 0,  1,  1,  2,  7],
            [49,  0,  7,  2,  1]]),
    values=tensor([ 40,  20, 100,  60,  80]),
    shape=(10, 50), nnz=5)
    >>> w = torch.arange(1, len(rowA)+1)
    >>> A + A(w)
    SparseMatrix(indices=tensor([[ 0,  1,  1,  2,  7],
            [49,  0,  7,  2,  1]]),
    values=tensor([21, 12, 53, 34, 45]),
    shape=(10, 50), nnz=5)

    Case 2: Add two matrices of different sparsity structure

    >>> rowB = torch.tensor([1, 9, 2, 7, 1, 1, 0])
    >>> colB = torch.tensor([0, 1, 2, 1, 7, 11, 15])
    >>> valB = torch.tensor([1, 2, 3, 4, 5, 6])
    >>> B = SparseMatrix(rowB, colB, valB, shape=(10, 50))
    >>> A + B
    SparseMatrix(indices=tensor([[ 0, 1, 1, 1, 2, 7, 9],
            [49, 0, 7, 11, 2, 1, 1]]),
    values=tensor([20, 11, 55,  6, 33, 44,  2]),
    shape=(10, 50), nnz=7)
    """
    if isinstance(A, SparseMatrix) and isinstance(B, SparseMatrix):
        assert A.shape == B.shape, 'The shape of sparse matrix A {} and' \
        ' B {} are expected to match'.format(A.shape, B.shape)
        C = A.adj + B.adj
        return SparseMatrix(C.indices()[0], C.indices()[1], C.values(), C.shape)
    raise RuntimeError('Elementwise addition between {} and {} is not ' \
                       'supported.'.format(type(A), type(B)))

def sub(A, B):
    """Elementwise subtraction.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    B : SparseMatrix
        Sparse matrix

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    >>> rowA = torch.tensor([1, 0, 2, 7, 1])
    >>> colA = torch.tensor([0, 49, 2, 1, 7])
    >>> valA = torch.tensor([10, 20, 30, 40, 50])
    >>> A = SparseMatrix(rowA, colA, valA, shape=(10, 50))
    >>> rowB = torch.tensor([1, 9, 2, 7, 1, 1])
    >>> colB = torch.tensor([0, 1, 2, 1, 7, 11])
    >>> valB = torch.tensor([1, 2, 3, 4, 5, 6])
    >>> B = SparseMatrix(rowB, colB, valB, shape=(10, 50))
    >>> A - B
    SparseMatrix(indices=tensor([[ 0, 1, 1, 1, 2, 7, 9],
            [49, 0, 7, 11, 2, 1, 1]]),
    values=tensor([20,  9, 45, -6, 27, 36, -2]),
    shape=(10, 50), nnz=7
    """
    if isinstance(A, SparseMatrix) and isinstance(B, SparseMatrix):
        assert A.shape == B.shape, 'The shape of sparse matrix A {} and' \
        ' B {} are expected to match.'.format(A.shape, B.shape)
        C = A.adj - B.adj
        return SparseMatrix(C.indices()[0], C.indices()[1], C.values(), C.shape)
    raise RuntimeError('Elementwise subtraction between {} and {} is not ' \
                       'supported.'.format(type(A), type(B)))

def mul(A, B):
    """Elementwise multiplication.

    Parameters
    ----------
    A : SparseMatrix or scalar
        Sparse matrix or scalar value
    B : SparseMatrix or scalar
        Sparse matrix or scalar value.

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    Case 1: Elementwise multiplication between two sparse matrices

    >>> rowA = torch.tensor([1, 0, 2, 7, 1])
    >>> colA = torch.tensor([0, 49, 2, 1, 7])
    >>> valA = torch.tensor([10, 20, 30, 40, 50])
    >>> A = SparseMatrix(rowA, colA, valA, shape=(10, 50))
    >>> rowB = torch.tensor([1, 9, 2, 7, 1, 1])
    >>> colB = torch.tensor([0, 1, 2, 1, 7, 11])
    >>> valB = torch.tensor([1, 2, 3, 4, 5, 6])
    >>> B = SparseMatrix(rowB, colB, valB, shape=(10, 50))
    >>> A * B
    SparseMatrix(indices=tensor([[1, 1, 2, 7],
            [0, 7, 2, 1]]),
    values=tensor([ 10, 250,  90, 160]),
    shape=(10, 50), nnz=4)

    Case 2: Elementwise multiplication between sparse matrix and scalar

    >>> v_scalar = 2.5
    >>> A * v_scalar
    SparseMatrix(indices=tensor([[ 0,  1,  1,  2,  7],
            [49,  0,  7,  2,  1]]),
    values=tensor([ 50.,  25., 125.,  75., 100.]),
    shape=(8, 50), nnz=5)
    >>> v_scalar * A
    SparseMatrix(indices=tensor([[ 0,  1,  1,  2,  7],
            [49,  0,  7,  2,  1]]),
    values=tensor([ 50.,  25., 125.,  75., 100.]),
    shape=(8, 50), nnz=5)
    """
    if isinstance(A, SparseMatrix) and isinstance(B, SparseMatrix):
        assert A.shape == B.shape, 'The shape of sparse matrix A {} and' \
        ' B {} are expected to match.'.format(A.shape, B.shape)
    A = A.adj if isinstance(A, SparseMatrix) else A
    B = B.adj if isinstance(B, SparseMatrix) else B
    C = A * B
    return SparseMatrix(C.indices()[0], C.indices()[1], C.values(), C.shape)

def div(A, B):
    """Elementwise division.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    B : SparseMatrix or scalar
        Sparse matrix or scalar value.

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    Case 1: Elementwise division between two matrices of same sparsity (matrices
            with different sparsity is not supported)

    >>> rowA = torch.tensor([1, 0, 2, 7, 1])
    >>> colA = torch.tensor([0, 49, 2, 1, 7])
    >>> valA = torch.tensor([10, 20, 30, 40, 50])
    >>> A = SparseMatrix(rowA, colA, valA, shape=(10, 50))
    >>> w = torch.arange(1, len(rowA)+1)
    >>> A/A(w)
    SparseMatrix(indices=tensor([[ 0,  1,  1,  2,  7],
            [49,  0,  7,  2,  1]]),
    values=tensor([20.0000,  5.0000, 16.6667,  7.5000,  8.0000]),
    shape=(8, 50), nnz=5)

    Case 2: Elementwise multiplication between sparse matrix and scalar

    >>> A / v_scalar
    SparseMatrix(indices=tensor([[ 0, 1, 1, 2, 7],
            [49, 0, 7, 2, 1]]),
    values=tensor([ 8., 4., 20., 12., 16.]),
    shape=(8, 50), nnz=5)
    """
    if isinstance(A, SparseMatrix) and isinstance(B, SparseMatrix):
        # same sparsity structure
        if torch.equal(A.indices("COO"), B.indices("COO")):
            return SparseMatrix(A.row, A.col, A.val / B.val, A.shape)
        raise ValueError('Division between matrices of different sparsity is not supported')
    C = A.adj/B
    return SparseMatrix(C.indices()[0], C.indices()[1], C.values(), C.shape)

def rdiv(A, B):
    """Elementwise division.

    Parameters
    ----------
    A : scalar
        scalar value
    B : SparseMatrix
        Sparse matrix
    """
    raise RuntimeError('Elementwise division between {} and {} is not ' \
                       'supported.'.format(type(A), type(B)))

def power(A, B):
    """Elementwise power operation.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    B : scalar
        scalar value.

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    >>> rowA = torch.tensor([1, 0, 2, 7, 1])
    >>> colA = torch.tensor([0, 49, 2, 1, 7])
    >>> valA = torch.tensor([10, 20, 30, 40, 50])
    >>> A = SparseMatrix(rowA, colA, valA, shape=(10, 50))
    >>> pow(A, 2.5)
    SparseMatrix(indices=tensor([[ 0,  1,  1,  2,  7],
            [49,  0,  7,  2,  1]]),
    values=tensor([ 1788.8544, 316.2278, 17677.6699,  4929.5029, 10119.2881]),
    shape=(8, 50), nnz=5)
    """
    if isinstance(B, SparseMatrix):
        raise RuntimeError('Power operation between two sparse matrices is not supported')
    return SparseMatrix(A.row, A.col, torch.pow(A.val, B))

def rpower(A, B):
    """Elementwise power operation.

    Parameters
    ----------
    A : scalar
        scalar value.
    B : SparseMatrix
        Sparse matrix.
    """
    raise RuntimeError('Power operation between {} and {} is not ' \
                       'supported.'.format(type(A), type(B)))

SparseMatrix.__add__ = add
SparseMatrix.__radd__ = add
SparseMatrix.__sub__ = sub
SparseMatrix.__rsub__ = sub
SparseMatrix.__mul__ = mul
SparseMatrix.__rmul__ = mul
SparseMatrix.__truediv__ = div
SparseMatrix.__rtruediv__ = rdiv
SparseMatrix.__pow__ = power
SparseMatrix.__rpow__ = rpower
