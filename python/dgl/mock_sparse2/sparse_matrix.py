"""DGL sparse matrix module."""
import os
import torch
from typing import List, Optional, Tuple

from .._ffi.base import dgl_sparse_loaded

assert dgl_sparse_loaded

# We may not need to use this constructor.
_C_SparseMatrixConstructor = torch.classes.dgl_sparse.SparseMatrix

class SparseMatrix:
    r"""Class for sparse matrix.

    Parameters
    ----------
    c_sparse_matrix : torch.ScriptObject
        C++ SparseMatrix object
    """
    def __init__(self, c_sparse_matrix: torch.ScriptObject):
        self.c_sparse_matrix = c_sparse_matrix

    @property
    def val(self) -> torch.Tensor:
        """Get the values of the nonzero elements.

        Returns
        -------
        torch.Tensor
            Values of the nonzero elements
        """
        return self.c_sparse_matrix.val()

    @property
    def shape(self) -> List[int]:
        """Shape of the sparse matrix.

        Returns
        -------
        List[int]
            The shape of the matrix
        """
        return self.c_sparse_matrix.shape()
    
    def coo(self) -> Tuple[torch.Tensor, ...] :
        """Get the coordinate (COO) representation of the sparse matrix.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple of tensors containing row, column coordinates and values.
        """
        return self.c_sparse_matrix.coo()

# TODO To make the docstring complete, we need to define SparseMatrix.__repr__
def create_from_coo(
    row: torch.Tensor,
    col: torch.Tensor,
    val: Optional[torch.Tensor] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> SparseMatrix:
    """Create a sparse matrix from row and column coordinates.

    Parameters
    ----------
    row : tensor
        The row indices of shape (nnz).
    col : tensor
        The column indices of shape (nnz).
    val : tensor, optional
        The values of shape (nnz) or (nnz, D). If None, it will be a tensor of shape (nnz)
        filled by 1.
    shape : tuple[int, int], optional
        If not specified, it will be inferred from :attr:`row` and :attr:`col`, i.e.,
        (row.max() + 1, col.max() + 1). Otherwise, :attr:`shape` should be no smaller
        than this.

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------

    Case1: Sparse matrix with row and column indices without values.

    >>> src = torch.tensor([1, 1, 2])
    >>> dst = torch.tensor([2, 4, 3])
    >>> A = create_from_coo(src, dst)
    >>> A
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([1., 1., 1.]),
                 shape=(3, 5), nnz=3)
    >>> # Specify shape
    >>> A = create_from_coo(src, dst, shape=(5, 5))
    >>> A
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([1., 1., 1.]),
                 shape=(5, 5), nnz=3)

    Case2: Sparse matrix with scalar/vector values. Following example is with
    vector data.

    >>> val = torch.tensor([[1, 1], [2, 2], [3, 3]])
    >>> A = create_from_coo(src, dst, val)
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([[1, 1],
                                [2, 2],
                                [3, 3]]),
                 shape=(3, 5), nnz=3)
    """
    return SparseMatrix(
        torch.ops.dgl_sparse.create_from_coo(row, col, val, shape)
    )

