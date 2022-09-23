"""dgl reduce operators for sparse matrix module."""
from typing import Optional
import torch

from .sp_matrix import SparseMatrix


def reduce(A: SparseMatrix, dim: Optional[int]=None, rtype: str = "sum"):
    """Compute the reduction of non-zero values in sparse matrix A along
    the given dimension :attr:`dim`.

    If :attr:`dim` is None, it reduces all the elements in the sparse
    matrix. Otherwise, it reduces on the row (``dim=0``) or column (``dim=1``)
    dimension, producing a tensor of shape ``(A.shape[1], ) + A.val.shape[:1]``
    or ``(A.shape[0],) + A.val.shape[:1]``.

    The reduction does not count zero values. If the row or column to be
    reduced does not have any non-zero value, the result will be 0.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    dim : int, optional
        The dimension to reduce.
    rtype: str
        Reduction type, one of ['sum', 'smin', 'smax', 'smean']

    Returns
    ----------
    Tensor
        Reduced tensor

    Examples
    ----------

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([1, 1, 2])
    >>> A = create_from_coo(row, col, val, shape=(4, 3))
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
    >>> val = torch.tensor([[1, 2], [2, 1], [2, 2]])
    >>> A = create_from_coo(row, col, val, shape=(4, 3))
    >>> print(A.reduce(reduce='sum'))
    tensor([5, 5])
    >>> print(A.reduce(0, 'sum'))
    tensor([[3, 3], [0, 0], [2, 2]])
    >>> print(A.reduce(1, 'smin'))
    tensor([[1, 2], [2, 1], [0, 0], [0, 0]])
    >>> print(A.reduce(0, 'smean'))
    tensor([[1, 1], [0, 0], [2, 2]])
    """
    if dim is not None and not isinstance(dim, int):
        raise ValueError(f"Reduce dimension should be int but got {dim}")

    if dim is None:
        if rtype == "sum":
            return torch.sum(A.val, dim=0)
        if rtype == "smax":
            return torch.amax(A.val, dim=0)
        if rtype == "smin":
            return torch.amin(A.val, dim=0)
        if rtype == "smean":
            return torch.mean(A.val, dim=0, dtype=torch.float64).to(A.val.dtype)

    if dim == 0:
        index = A.col
        reduced_shape = (A.shape[1],) + A.val.shape[1:]
        reduced = torch.zeros(reduced_shape, dtype=A.val.dtype, device=A.device)
    else:
        index = A.row
        reduced_shape = (A.shape[0],) + A.val.shape[1:]
        reduced = torch.zeros(reduced_shape, dtype=A.val.dtype, device=A.device)

    if rtype in ("smax", "smin"):
        rtype = "a" + rtype[1:]

    if rtype == "smean":
        rtype = "mean"

    if len(A.val.shape) > 1:
        index = torch.unsqueeze(index, 1)
        index = index.repeat([1, A.val.shape[1]])
    reduced = reduced.scatter_reduce(
        0, index, A.val, reduce=rtype, include_self=False
    )
    return reduced


def sum(A: SparseMatrix, dim: Optional[int]=None):  # pylint: disable=W0622
    """Compute the sum of non-zero values in sparse matrix A along
    the given dimension :attr:`dim`.

    If :attr:`dim` is None, it reduces all the elements in the sparse matrix.
    Otherwise, it reduces on the row (``dim=0``) or column (``dim=1``) dimension,
    producing a tensor of shape ``(A.shape[1], ) + A.val.shape[:1]`` or
    ``(A.shape[0],) + A.val.shape[:1]``.

    Parameters
    ----------
    dim : int, optional
        The dimension to reduce.

    Returns
    ----------
    Tensor
        Reduced tensor

    Examples
    ----------

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([1, 1, 2])
    >>> A = create_from_coo(row, col, val, shape=(4, 3))
    >>> print(A.sum())
    tensor(4)
    >>> print(A.sum(0))
    tensor([2, 0, 2])
    >>> print(A.sum(1))
    tensor([1, 3, 0, 0])

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([[1, 2], [2, 1], [2, 2]])
    >>> A = create_from_coo(row, col, val, shape=(4, 3))
    >>> print(A.sum())
    tensor([5, 5])
    >>> print(A.sum(0))
    tensor([[3, 3], [0, 0], [2, 2]])
    """
    return A.reduce(dim, rtype="sum")


def smax(A: SparseMatrix, dim: Optional[int]=None):
    """Compute the maximum of non-zero values in sparse matrix A along
    the given dimension :attr:`dim`.

    If :attr:`dim` is None, it reduces all the elements in the sparse matrix.
    Otherwise, it reduces on the row (``dim=0``) or column (``dim=1``) dimension,
    producing a tensor of shape ``(A.shape[1], ) + A.val.shape[:1]`` or
    ``(A.shape[0],) + A.val.shape[:1]``.

    The reduction does not count zero values. If the row or column to be
    reduced does not have any non-zero value, the result will be 0.

    Parameters
    ----------
    dim : int, optional
        The dimension to reduce.

    Returns
    ----------
    Tensor
        Reduced tensor

    Examples
    ----------

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([1, 1, 2])
    >>> A = create_from_coo(row, col, val, shape=(4, 3))
    >>> print(A.smax())
    tensor(2)
    >>> print(A.smax(0))
    tensor([1, 0, 2])
    >>> print(A.smax(1))
    tensor([1, 2, 0, 0])

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([[1, 2], [2, 1], [2, 2]])
    >>> A = create_from_coo(row, col, val, shape=(4, 3))
    >>> print(A.smax())
    tensor([2, 2])
    >>> print(A.smax(0))
    tensor([[2, 2], [0, 0], [2, 2]])
    >>> print(A.smax(1))
    tensor([[1, 2], [2, 2], [0, 0], [0, 0]])
    """
    return A.reduce(dim, rtype="smax")


def smin(A: SparseMatrix, dim: Optional[int]=None):
    """Compute the minimum of non-zero values in sparse matrix A along
    the given dimension :attr:`dim`.

    If :attr:`dim` is None, it reduces all the elements in the sparse matrix.
    Otherwise, it reduces on the row (``dim=0``) or column (``dim=1``) dimension,
    producing a tensor of shape ``(A.shape[1], ) + A.val.shape[:1]`` or
    ``(A.shape[0],) + A.val.shape[:1]``.

    The reduction does not count zero values. If the row or column to be reduced
    does not have any non-zero value, the result will be 0.

    Parameters
    ----------
    dim : int, optional
        The dimension to reduce.

    Returns
    ----------
    Tensor
        Reduced tensor

    Example
    ----------

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([1, 1, 2])
    >>> A = create_from_coo(row, col, val, shape=(4, 3))
    >>> print(A.smin())
    tensor(1)
    >>> print(A.smin(0))
    tensor([1, 0, 2])
    >>> print(A.smin(1))
    tensor([1, 1, 0, 0])

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([[1, 2], [2, 1], [2, 2]])
    >>> A = create_from_coo(row, col, val, shape=(4, 3))
    >>> print(A.smin())
    tensor([1, 1])
    >>> print(A.smin(0))
    tensor([[1, 1], [0, 0], [2, 2]])
    >>> print(A.smin(1))
    tensor([[1, 2], [2, 1], [0, 0], [0, 0]])
    """
    return A.reduce(dim, rtype="smin")


def smean(A: SparseMatrix, dim: Optional[int]=None):
    """Compute the mean of non-zero values in sparse matrix A along
    the given dimension :attr:`dim`.

    If :attr:`dim` is None, it reduces all the elements in the sparse matrix.
    Otherwise, it reduces on the row (``dim=0``) or column (``dim=1``) dimension,
    producing a tensor of shape ``(A.shape[1], ) + A.val.shape[:1]`` or
    ``(A.shape[0],) + A.val.shape[:1]``.

    The reduction does not count zero values. If the row or column to be reduced
    does not have any non-zero value, the result will be 0.

    Parameters
    ----------
    dim : int, optional
        The dimension to reduce.

    Returns
    ----------
    Tensor
        Reduced tensor

    Example
    ----------

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([1, 1, 2])
    >>> A = create_from_coo(row, col, val, shape=(4, 3))
    >>> print(A.smean())
    tensor(1)
    >>> print(A.smean(0))
    tensor([1, 0, 2])
    >>> print(A.smean(1))
    tensor([1, 1, 0, 0])

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([0, 0, 2])
    >>> val = torch.tensor([[1, 2], [2, 1], [2, 2]])
    >>> A = create_from_coo(row, col, val, shape=(4, 3))
    >>> print(A.smean())
    tensor([1, 1])
    >>> print(A.smean(0))
    tensor([[1, 1], [0, 0], [2, 2]])
    >>> print(A.smean(1))
    tensor([[1, 2], [2, 1], [0, 0], [0, 0]])
    """
    return A.reduce(dim, rtype="smean")


SparseMatrix.reduce = reduce
SparseMatrix.sum = sum
SparseMatrix.smax = smax
SparseMatrix.smin = smin
SparseMatrix.smean = smean
