"""DGL broadcast operator module."""

import operator

import torch

from .sparse_matrix import SparseMatrix, val_like


def sp_broadcast_v(A: SparseMatrix, v: torch.Tensor, op: str) -> SparseMatrix:
    """Broadcast operator for sparse matrix and vector.

    :attr:`v` is broadcasted to the shape of :attr:`A` and then the operator is
    applied on the non-zero values of :attr:`A`.

    There are two cases regarding the shape of v:

    1. :attr:`v` is a vector of shape ``(1, A.shape[1])`` or ``(A.shape[1])``.
    In this case, :attr:`v` is broadcasted on the row dimension of :attr:`A`.

    2. :attr:`v` is a vector of shape ``(A.shape[0], 1)``. In this case,
    :attr:`v` is broadcasted on the column dimension of :attr:`A`.

    If ``A.val`` takes shape ``(nnz, D)``, then :attr:`v` will be broadcasted on
    the ``D`` dimension.

    Parameters
    ----------
    A: SparseMatrix
        Sparse matrix
    v: torch.Tensor
        Vector
    op: str
        Operator in ["add", "sub", "mul", "truediv"]

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    >>> indices = torch.tensor([[1, 0, 2], [0, 3, 2]])
    >>> val = torch.tensor([10, 20, 30])
    >>> A = dglsp.spmatrix(indices, val, shape=(3, 4))
    >>> v = torch.tensor([1, 2, 3, 4])
    >>> dglsp.sp_broadcast_v(A, v, "add")
    SparseMatrix(indices=tensor([[1, 0, 2],
                                 [0, 3, 2]]),
                 values=tensor([11, 24, 33]),
                 shape=(3, 4), nnz=3)

    >>> v = torch.tensor([1, 2, 3]).view(-1, 1)
    >>> dglsp.sp_broadcast_v(A, v, "add")
    SparseMatrix(indices=tensor([[1, 0, 2],
                                 [0, 3, 2]]),
                 values=tensor([12, 21, 33]),
                 shape=(3, 4), nnz=3)

    >>> indices = torch.tensor([[1, 0, 2], [0, 3, 2]])
    >>> val = torch.tensor([[10, 20], [30, 40], [50, 60]])
    >>> A = dglsp.spmatrix(indices, val, shape=(3, 4))
    >>> v = torch.tensor([1, 2, 3]).view(-1, 1)
    >>> dglsp.sp_broadcast_v(A, v, "sub")
    SparseMatrix(indices=tensor([[1, 0, 2],
                                 [0, 3, 2]]),
                 values=tensor([[ 8, 18],
                                [29, 39],
                                [47, 57]]),
                 shape=(3, 4), nnz=3, val_size=(2,))
    """
    op = getattr(operator, op)
    if v.dim() == 1:
        v = v.view(1, -1)

    shape_error_message = (
        f"Dimension mismatch for broadcasting. Got A.shape = {A.shape} and"
        f"v.shape = {v.shape}."
    )
    assert v.dim() <= 2 and (1 in v.shape), shape_error_message
    broadcast_dim = None
    # v can be broadcasted to A if exactly one dimension of v is 1 and the other
    # is the same as A.
    for d, (dim1, dim2) in enumerate(zip(A.shape, v.shape)):
        assert dim2 in (1, dim1), shape_error_message
        if dim1 != dim2:
            assert broadcast_dim is None, shape_error_message
            broadcast_dim = d

    # A and v has the same shape of (1, *) or (*, 1).
    if broadcast_dim is None:
        broadcast_dim = 0 if A.shape[0] == 1 else 1

    if broadcast_dim == 0:
        v = v.view(-1)[A.col]
    else:
        v = v.view(-1)[A.row]
    if A.val.dim() > 1:
        v = v.view(-1, 1)
    ret_val = op(A.val, v)
    return val_like(A, ret_val)


def sp_add_v(A: SparseMatrix, v: torch.Tensor) -> SparseMatrix:
    """Broadcast addition for sparse matrix and vector.

    See the definition of :func:`sp_broadcast_v` for details.
    """
    return sp_broadcast_v(A, v, "add")


def sp_sub_v(A: SparseMatrix, v: torch.Tensor) -> SparseMatrix:
    """Broadcast substraction for sparse matrix and vector.

    See the definition of :func:`sp_broadcast_v` for details.
    """
    return sp_broadcast_v(A, v, "sub")


def sp_mul_v(A: SparseMatrix, v: torch.Tensor) -> SparseMatrix:
    """Broadcast multiply for sparse matrix and vector.

    See the definition of :func:`sp_broadcast_v` for details.
    """
    return sp_broadcast_v(A, v, "mul")


def sp_div_v(A: SparseMatrix, v: torch.Tensor) -> SparseMatrix:
    """Broadcast division for sparse matrix and vector.

    See the definition of :func:`sp_broadcast_v` for details.
    """
    return sp_broadcast_v(A, v, "truediv")
