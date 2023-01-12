"""DGL sparse matrix reduce operators"""
from typing import Optional

import torch

from .sparse_matrix import SparseMatrix


def reduce(A: SparseMatrix, dim: Optional[int] = None, rtype: str = "sum"):
    """Compute the reduction of non-zero values in sparse matrix A along
    the given dimension :attr:`dim`.

    The reduction does not count zero values. If the row or column to be
    reduced does not have any non-zero value, the result will be 0.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    dim : int, optional
        The dimension to reduce.  Must be either 0 (by rows) or 1 (by columns)
        or None (on all non-zero entries).

        If :attr:`dim` is None, it reduces all the elements in the sparse
        matrix. Otherwise, it reduces on the row (``dim=0``) or column
        (``dim=1``) dimension, producing a tensor of shape
        ``(A.shape[1],) + A.val.shape[1:]`` or
        ``(A.shape[0],) + A.val.shape[1:]``.
    rtype: str, optional
        Reduction type, one of ``['sum', 'smin', 'smax', 'smean', 'sprod']``,
        representing taking the sum, minimum, maximum, mean, and product of the
        non-zero entries.

    Returns
    ----------
    Tensor
        Reduced tensor

    Examples
    ----------

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([1, 1, 2])
    >>> A = dglsp.from_coo(row, col, val, shape=(4, 3))
    >>> print(A.reduce(rtype='sum'))
    tensor(4)
    >>> print(A.reduce(0, 'sum'))
    tensor([2, 0, 2])
    >>> print(A.reduce(1, 'sum'))
    tensor([1, 3, 0, 0])
    >>> print(A.reduce(0, 'smax'))
    tensor([1, 0, 2])
    >>> print(A.reduce(1, 'smin'))
    tensor([1, 1, 0, 0])

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([[1., 2.], [2., 1.], [2., 2.]])
    >>> A = dglsp.from_coo(row, col, val, shape=(4, 3))
    >>> print(A.reduce(rtype='sum'))
    tensor([5., 5.])
    >>> print(A.reduce(0, 'sum'))
    tensor([[3., 3.],
            [0., 0.],
            [2., 2.]])
    >>> print(A.reduce(1, 'smin'))
    tensor([[1., 2.],
            [2., 1.],
            [0., 0.],
            [0., 0.]])
    >>> print(A.reduce(0, 'smean'))
    tensor([[1.5000, 1.5000],
            [0.0000, 0.0000],
            [2.0000, 2.0000]])
    """
    return torch.ops.dgl_sparse.reduce(A.c_sparse_matrix, rtype, dim)


# pylint: disable=W0622
def sum(A: SparseMatrix, dim: Optional[int] = None):
    """Compute the sum of non-zero values in sparse matrix A along
    the given dimension :attr:`dim`.

    Parameters
    ----------
    dim : int, optional
        The dimension to reduce.  Must be either 0 (by rows) or 1 (by columns)
        or None (on all non-zero entries).

        If :attr:`dim` is None, it reduces all the elements in the sparse
        matrix. Otherwise, it reduces on the row (``dim=0``) or column
        (``dim=1``) dimension, producing a tensor of shape
        ``(A.shape[1],) + A.val.shape[1:]`` or
        ``(A.shape[0],) + A.val.shape[1:]``.

    Returns
    ----------
    Tensor
        Reduced tensor

    Examples
    ----------

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([1, 1, 2])
    >>> A = dglsp.from_coo(row, col, val, shape=(4, 3))
    >>> print(A.sum())
    tensor(4)
    >>> print(A.sum(0))
    tensor([2, 0, 2])
    >>> print(A.sum(1))
    tensor([1, 3, 0, 0])

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([[1, 2], [2, 1], [2, 2]])
    >>> A = dglsp.from_coo(row, col, val, shape=(4, 3))
    >>> print(A.sum())
    tensor([5, 5])
    >>> print(A.sum(0))
    tensor([[3, 3],
            [0, 0],
            [2, 2]])
    """
    return torch.ops.dgl_sparse.sum(A.c_sparse_matrix, dim)


def smax(A: SparseMatrix, dim: Optional[int] = None):
    """Compute the maximum of non-zero values in sparse matrix A along
    the given dimension :attr:`dim`.

    The reduction does not count zero values. If the row or column to be
    reduced does not have any non-zero value, the result will be 0.

    Parameters
    ----------
    dim : int, optional
        The dimension to reduce.  Must be either 0 (by rows) or 1 (by columns)
        or None (on all non-zero entries).

        If :attr:`dim` is None, it reduces all the elements in the sparse
        matrix. Otherwise, it reduces on the row (``dim=0``) or column
        (``dim=1``) dimension, producing a tensor of shape
        ``(A.shape[1],) + A.val.shape[1:]`` or
        ``(A.shape[0],) + A.val.shape[1:]``.

    Returns
    ----------
    Tensor
        Reduced tensor

    Examples
    ----------

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([1, 1, 2])
    >>> A = dglsp.from_coo(row, col, val, shape=(4, 3))
    >>> print(A.smax())
    tensor(2)
    >>> print(A.smax(0))
    tensor([1, 0, 2])
    >>> print(A.smax(1))
    tensor([1, 2, 0, 0])

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([[1, 2], [2, 1], [2, 2]])
    >>> A = dglsp.from_coo(row, col, val, shape=(4, 3))
    >>> print(A.smax())
    tensor([2, 2])
    >>> print(A.smax(1))
    tensor([[1, 2],
            [2, 2],
            [0, 0],
            [0, 0]])
    """
    return torch.ops.dgl_sparse.smax(A.c_sparse_matrix, dim)


def smin(A: SparseMatrix, dim: Optional[int] = None):
    """Compute the minimum of non-zero values in sparse matrix A along
    the given dimension :attr:`dim`.

    The reduction does not count zero values. If the row or column to be reduced
    does not have any non-zero value, the result will be 0.

    Parameters
    ----------
    dim : int, optional
        The dimension to reduce.  Must be either 0 (by rows) or 1 (by columns)
        or None (on all non-zero entries).

        If :attr:`dim` is None, it reduces all the elements in the sparse
        matrix. Otherwise, it reduces on the row (``dim=0``) or column
        (``dim=1``) dimension, producing a tensor of shape
        ``(A.shape[1],) + A.val.shape[1:]`` or
        ``(A.shape[0],) + A.val.shape[1:]``.

    Returns
    ----------
    Tensor
        Reduced tensor

    Example
    ----------

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([1, 1, 2])
    >>> A = dglsp.from_coo(row, col, val, shape=(4, 3))
    >>> print(A.smin())
    tensor(1)
    >>> print(A.smin(0))
    tensor([1, 0, 2])
    >>> print(A.smin(1))
    tensor([1, 1, 0, 0])

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([[1, 2], [2, 1], [2, 2]])
    >>> A = dglsp.from_coo(row, col, val, shape=(4, 3))
    >>> print(A.smin())
    tensor([1, 1])
    >>> print(A.smin(0))
    tensor([[1, 1],
            [0, 0],
            [2, 2]])
    >>> print(A.smin(1))
    tensor([[1, 2],
            [2, 1],
            [0, 0],
            [0, 0]])
    """
    return torch.ops.dgl_sparse.smin(A.c_sparse_matrix, dim)


def smean(A: SparseMatrix, dim: Optional[int] = None):
    """Compute the mean of non-zero values in sparse matrix A along
    the given dimension :attr:`dim`.

    The reduction does not count zero values. If the row or column to be reduced
    does not have any non-zero value, the result will be 0.

    Parameters
    ----------
    dim : int, optional
        The dimension to reduce.  Must be either 0 (by rows) or 1 (by columns)
        or None (on all non-zero entries).

        If :attr:`dim` is None, it reduces all the elements in the sparse
        matrix. Otherwise, it reduces on the row (``dim=0``) or column
        (``dim=1``) dimension, producing a tensor of shape
        ``(A.shape[1],) + A.val.shape[1:]`` or
        ``(A.shape[0],) + A.val.shape[1:]``.

    Returns
    ----------
    Tensor
        Reduced tensor

    Example
    ----------

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([1., 1., 2.])
    >>> A = dglsp.from_coo(row, col, val, shape=(4, 3))
    >>> print(A.smean())
    tensor(1.3333)
    >>> print(A.smean(0))
    tensor([1., 0., 2.])
    >>> print(A.smean(1))
    tensor([1.0000, 1.5000, 0.0000, 0.0000])

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([[1., 2.], [2., 1.], [2., 2.]])
    >>> A = dglsp.from_coo(row, col, val, shape=(4, 3))
    >>> print(A.smean())
    tensor([1.6667, 1.6667])
    >>> print(A.smean(0))
    tensor([[1.5000, 1.5000],
            [0.0000, 0.0000],
            [2.0000, 2.0000]])
    >>> print(A.smean(1))
    tensor([[1.0000, 2.0000],
            [2.0000, 1.5000],
            [0.0000, 0.0000],
            [0.0000, 0.0000]])
    """
    return torch.ops.dgl_sparse.smean(A.c_sparse_matrix, dim)


def sprod(A: SparseMatrix, dim: Optional[int] = None):
    """Compute the product of non-zero values in sparse matrix A along
    the given dimension :attr:`dim`.

    The reduction does not count zero values. If the row or column to be reduced
    does not have any non-zero value, the result will be 0.

    Parameters
    ----------
    dim : int, optional
        The dimension to reduce.  Must be either 0 (by rows) or 1 (by columns)
        or None (on all non-zero entries).

        If :attr:`dim` is None, it reduces all the elements in the sparse
        matrix. Otherwise, it reduces on the row (``dim=0``) or column
        (``dim=1``) dimension, producing a tensor of shape
        ``(A.shape[1],) + A.val.shape[1:]`` or
        ``(A.shape[0],) + A.val.shape[1:]``.

    Returns
    ----------
    Tensor
        Reduced tensor

    Example
    ----------

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([1, 1, 2])
    >>> A = dglsp.from_coo(row, col, val, shape=(4, 3))
    >>> print(A.sprod())
    tensor(2)
    >>> print(A.sprod(0))
    tensor([1, 0, 2])
    >>> print(A.sprod(1))
    tensor([1, 2, 0, 0])

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([[1, 2], [2, 1], [2, 2]])
    >>> A = dglsp.from_coo(row, col, val, shape=(4, 3))
    >>> print(A.sprod())
    tensor([4, 4])
    >>> print(A.sprod(0))
    tensor([[2, 2],
            [0, 0],
            [2, 2]])
    >>> print(A.sprod(1))
    tensor([[1, 2],
            [4, 2],
            [0, 0],
            [0, 0]])
    """
    return torch.ops.dgl_sparse.sprod(A.c_sparse_matrix, dim)


SparseMatrix.reduce = reduce
SparseMatrix.sum = sum
SparseMatrix.smax = smax
SparseMatrix.smin = smin
SparseMatrix.smean = smean
SparseMatrix.sprod = sprod
