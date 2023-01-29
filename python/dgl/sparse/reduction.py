"""DGL sparse matrix reduce operators"""
# pylint: disable=W0622

from typing import Optional

import torch

from .sparse_matrix import SparseMatrix


def reduce(input: SparseMatrix, dim: Optional[int] = None, rtype: str = "sum"):
    """Computes the reduction of non-zero values of the :attr:`input` sparse
    matrix along the given dimension :attr:`dim`.

    The reduction does not count zero elements. If the row or column to be
    reduced does not have any non-zero elements, the result will be 0.

    Parameters
    ----------
    input : SparseMatrix
        The input sparse matrix
    dim : int, optional
        The dimension to reduce, must be either 0 (by rows) or 1 (by columns)
        or None (on both rows and columns simultaneously)

        If :attr:`dim` is None, it reduces both the rows and the columns
        in the sparse matrix, producing a tensor of shape
        ``input.val.shape[1:]``. Otherwise, it reduces on the row (``dim=0``)
        or column (``dim=1``) dimension, producing a tensor of shape
        ``(input.shape[1],) + input.val.shape[1:]`` or
        ``(input.shape[0],) + input.val.shape[1:]``.
    rtype: str, optional
        Reduction type, one of ``['sum', 'smin', 'smax', 'smean', 'sprod']``,
        representing taking the sum, minimum, maximum, mean, and product of the
        non-zero elements

    Returns
    ----------
    torch.Tensor
        Reduced tensor

    Examples
    ----------

    Case1: scalar-valued sparse matrix

    >>> indices = torch.tensor([[0, 1, 1], [0, 0, 2]])
    >>> val = torch.tensor([1, 1, 2])
    >>> A = dglsp.spmatrix(indices, val, shape=(4, 3))
    >>> dglsp.reduce(A, rtype='sum')
    tensor(4)
    >>> dglsp.reduce(A, 0, 'sum')
    tensor([2, 0, 2])
    >>> dglsp.reduce(A, 1, 'sum')
    tensor([1, 3, 0, 0])
    >>> dglsp.reduce(A, 0, 'smax')
    tensor([1, 0, 2])
    >>> dglsp.reduce(A, 1, 'smin')
    tensor([1, 1, 0, 0])

    Case2: vector-valued sparse matrix

    >>> indices = torch.tensor([[0, 1, 1], [0, 0, 2]])
    >>> val = torch.tensor([[1., 2.], [2., 1.], [2., 2.]])
    >>> A = dglsp.spmatrix(indices, val, shape=(4, 3))
    >>> dglsp.reduce(A, rtype='sum')
    tensor([5., 5.])
    >>> dglsp.reduce(A, 0, 'sum')
    tensor([[3., 3.],
            [0., 0.],
            [2., 2.]])
    >>> dglsp.reduce(A, 1, 'smin')
    tensor([[1., 2.],
            [2., 1.],
            [0., 0.],
            [0., 0.]])
    >>> dglsp.reduce(A, 0, 'smean')
    tensor([[1.5000, 1.5000],
            [0.0000, 0.0000],
            [2.0000, 2.0000]])
    """
    return torch.ops.dgl_sparse.reduce(input.c_sparse_matrix, rtype, dim)


def sum(input: SparseMatrix, dim: Optional[int] = None):
    """Computes the sum of non-zero values of the :attr:`input` sparse matrix
    along the given dimension :attr:`dim`.

    Parameters
    ----------
    input : SparseMatrix
        The input sparse matrix
    dim : int, optional
        The dimension to reduce, must be either 0 (by rows) or 1 (by columns)
        or None (on both rows and columns simultaneously)

        If :attr:`dim` is None, it reduces both the rows and the columns
        in the sparse matrix, producing a tensor of shape
        ``input.val.shape[1:]``. Otherwise, it reduces on the row (``dim=0``)
        or column (``dim=1``) dimension, producing a tensor of shape
        ``(input.shape[1],) + input.val.shape[1:]`` or
        ``(input.shape[0],) + input.val.shape[1:]``.

    Returns
    ----------
    torch.Tensor
        Reduced tensor

    Examples
    ----------

    Case1: scalar-valued sparse matrix

    >>> indices = torch.tensor([[0, 1, 1], [0, 0, 2]])
    >>> val = torch.tensor([1, 1, 2])
    >>> A = dglsp.spmatrix(indices, val, shape=(4, 3))
    >>> dglsp.sum(A)
    tensor(4)
    >>> dglsp.sum(A, 0)
    tensor([2, 0, 2])
    >>> dglsp.sum(A, 1)
    tensor([1, 3, 0, 0])

    Case2: vector-valued sparse matrix

    >>> indices = torch.tensor([[0, 1, 1], [0, 0, 2]])
    >>> val = torch.tensor([[1, 2], [2, 1], [2, 2]])
    >>> A = dglsp.spmatrix(indices, val, shape=(4, 3))
    >>> dglsp.sum(A)
    tensor([5, 5])
    >>> dglsp.sum(A, 0)
    tensor([[3, 3],
            [0, 0],
            [2, 2]])
    """
    return torch.ops.dgl_sparse.sum(input.c_sparse_matrix, dim)


def smax(input: SparseMatrix, dim: Optional[int] = None):
    """Computes the maximum of non-zero values of the :attr:`input` sparse
    matrix along the given dimension :attr:`dim`.

    The reduction does not count zero values. If the row or column to be
    reduced does not have any non-zero value, the result will be 0.

    Parameters
    ----------
    input : SparseMatrix
        The input sparse matrix
    dim : int, optional
        The dimension to reduce, must be either 0 (by rows) or 1 (by columns)
        or None (on both rows and columns simultaneously)

        If :attr:`dim` is None, it reduces both the rows and the columns
        in the sparse matrix, producing a tensor of shape
        ``input.val.shape[1:]``. Otherwise, it reduces on the row (``dim=0``)
        or column (``dim=1``) dimension, producing a tensor of shape
        ``(input.shape[1],) + input.val.shape[1:]`` or
        ``(input.shape[0],) + input.val.shape[1:]``.

    Returns
    ----------
    torch.Tensor
        Reduced tensor

    Examples
    ----------

    Case1: scalar-valued sparse matrix

    >>> indices = torch.tensor([[0, 1, 1], [0, 0, 2]])
    >>> val = torch.tensor([1, 1, 2])
    >>> A = dglsp.spmatrix(indices, val, shape=(4, 3))
    >>> dglsp.smax(A)
    tensor(2)
    >>> dglsp.smax(A, 0)
    tensor([1, 0, 2])
    >>> dglsp.smax(A, 1)
    tensor([1, 2, 0, 0])

    Case2: vector-valued sparse matrix

    >>> indices = torch.tensor([[0, 1, 1], [0, 0, 2]])
    >>> val = torch.tensor([[1, 2], [2, 1], [2, 2]])
    >>> A = dglsp.spmatrix(indices, val, shape=(4, 3))
    >>> dglsp.smax(A)
    tensor([2, 2])
    >>> dglsp.smax(A, 1)
    tensor([[1, 2],
            [2, 2],
            [0, 0],
            [0, 0]])
    """
    return torch.ops.dgl_sparse.smax(input.c_sparse_matrix, dim)


def smin(input: SparseMatrix, dim: Optional[int] = None):
    """Computes the minimum of non-zero values of the :attr:`input` sparse
    matrix along the given dimension :attr:`dim`.

    The reduction does not count zero values. If the row or column to be reduced
    does not have any non-zero value, the result will be 0.

    Parameters
    ----------
    input : SparseMatrix
        The input sparse matrix
    dim : int, optional
        The dimension to reduce, must be either 0 (by rows) or 1 (by columns)
        or None (on both rows and columns simultaneously)

        If :attr:`dim` is None, it reduces both the rows and the columns
        in the sparse matrix, producing a tensor of shape
        ``input.val.shape[1:]``. Otherwise, it reduces on the row (``dim=0``)
        or column (``dim=1``) dimension, producing a tensor of shape
        ``(input.shape[1],) + input.val.shape[1:]`` or
        ``(input.shape[0],) + input.val.shape[1:]``.

    Returns
    ----------
    torch.Tensor
        Reduced tensor

    Examples
    ----------

    Case1: scalar-valued sparse matrix

    >>> indices = torch.tensor([[0, 1, 1], [0, 0, 2]])
    >>> val = torch.tensor([1, 1, 2])
    >>> A = dglsp.spmatrix(indices, val, shape=(4, 3))
    >>> dglsp.smin(A)
    tensor(1)
    >>> dglsp.smin(A, 0)
    tensor([1, 0, 2])
    >>> dglsp.smin(A, 1)
    tensor([1, 1, 0, 0])

    Case2: vector-valued sparse matrix

    >>> indices = torch.tensor([[0, 1, 1], [0, 0, 2]])
    >>> val = torch.tensor([[1, 2], [2, 1], [2, 2]])
    >>> A = dglsp.spmatrix(indices, val, shape=(4, 3))
    >>> dglsp.smin(A)
    tensor([1, 1])
    >>> dglsp.smin(A, 0)
    tensor([[1, 1],
            [0, 0],
            [2, 2]])
    >>> dglsp.smin(A, 1)
    tensor([[1, 2],
            [2, 1],
            [0, 0],
            [0, 0]])
    """
    return torch.ops.dgl_sparse.smin(input.c_sparse_matrix, dim)


def smean(input: SparseMatrix, dim: Optional[int] = None):
    """Computes the mean of non-zero values of the :attr:`input` sparse matrix
    along the given dimension :attr:`dim`.

    The reduction does not count zero values. If the row or column to be reduced
    does not have any non-zero value, the result will be 0.

    Parameters
    ----------
    input : SparseMatrix
        The input sparse matrix
    dim : int, optional
        The dimension to reduce, must be either 0 (by rows) or 1 (by columns)
        or None (on both rows and columns simultaneously)

        If :attr:`dim` is None, it reduces both the rows and the columns
        in the sparse matrix, producing a tensor of shape
        ``input.val.shape[1:]``. Otherwise, it reduces on the row (``dim=0``)
        or column (``dim=1``) dimension, producing a tensor of shape
        ``(input.shape[1],) + input.val.shape[1:]`` or
        ``(input.shape[0],) + input.val.shape[1:]``.

    Returns
    ----------
    torch.Tensor
        Reduced tensor

    Examples
    ----------

    Case1: scalar-valued sparse matrix

    >>> indices = torch.tensor([[0, 1, 1], [0, 0, 2]])
    >>> val = torch.tensor([1., 1., 2.])
    >>> A = dglsp.spmatrix(indices, val, shape=(4, 3))
    >>> dglsp.smean(A)
    tensor(1.3333)
    >>> dglsp.smean(A, 0)
    tensor([1., 0., 2.])
    >>> dglsp.smean(A, 1)
    tensor([1.0000, 1.5000, 0.0000, 0.0000])

    Case2: vector-valued sparse matrix

    >>> indices = torch.tensor([[0, 1, 1], [0, 0, 2]])
    >>> val = torch.tensor([[1., 2.], [2., 1.], [2., 2.]])
    >>> A = dglsp.spmatrix(indices, val, shape=(4, 3))
    >>> dglsp.smean(A)
    tensor([1.6667, 1.6667])
    >>> dglsp.smean(A, 0)
    tensor([[1.5000, 1.5000],
            [0.0000, 0.0000],
            [2.0000, 2.0000]])
    >>> dglsp.smean(A, 1)
    tensor([[1.0000, 2.0000],
            [2.0000, 1.5000],
            [0.0000, 0.0000],
            [0.0000, 0.0000]])
    """
    return torch.ops.dgl_sparse.smean(input.c_sparse_matrix, dim)


def sprod(input: SparseMatrix, dim: Optional[int] = None):
    """Computes the product of non-zero values of the :attr:`input` sparse
    matrix along the given dimension :attr:`dim`.

    The reduction does not count zero values. If the row or column to be reduced
    does not have any non-zero value, the result will be 0.

    Parameters
    ----------
    input : SparseMatrix
        The input sparse matrix
    dim : int, optional
        The dimension to reduce, must be either 0 (by rows) or 1 (by columns)
        or None (on both rows and columns simultaneously)

        If :attr:`dim` is None, it reduces both the rows and the columns
        in the sparse matrix, producing a tensor of shape
        ``input.val.shape[1:]``. Otherwise, it reduces on the row (``dim=0``)
        or column (``dim=1``) dimension, producing a tensor of shape
        ``(input.shape[1],) + input.val.shape[1:]`` or
        ``(input.shape[0],) + input.val.shape[1:]``.

    Returns
    ----------
    torch.Tensor
        Reduced tensor

    Examples
    ----------

    Case1: scalar-valued sparse matrix

    >>> indices = torch.tensor([[0, 1, 1], [0, 0, 2]])
    >>> val = torch.tensor([1, 1, 2])
    >>> A = dglsp.spmatrix(indices, val, shape=(4, 3))
    >>> dglsp.sprod(A)
    tensor(2)
    >>> dglsp.sprod(A, 0)
    tensor([1, 0, 2])
    >>> dglsp.sprod(A, 1)
    tensor([1, 2, 0, 0])

    Case2: vector-valued sparse matrix

    >>> indices = torch.tensor([[0, 1, 1], [0, 0, 2]])
    >>> val = torch.tensor([[1, 2], [2, 1], [2, 2]])
    >>> A = dglsp.spmatrix(indices, val, shape=(4, 3))
    >>> dglsp.sprod(A)
    tensor([4, 4])
    >>> dglsp.sprod(A, 0)
    tensor([[2, 2],
            [0, 0],
            [2, 2]])
    >>> dglsp.sprod(A, 1)
    tensor([[1, 2],
            [4, 2],
            [0, 0],
            [0, 0]])
    """
    return torch.ops.dgl_sparse.sprod(input.c_sparse_matrix, dim)


SparseMatrix.reduce = reduce
SparseMatrix.sum = sum
SparseMatrix.smax = smax
SparseMatrix.smin = smin
SparseMatrix.smean = smean
SparseMatrix.sprod = sprod
