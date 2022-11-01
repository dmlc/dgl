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
        tensor
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

# TODO Add docstring
def create_from_coo(
    row: torch.Tensor,
    col: torch.Tensor,
    val: Optional[torch.Tensor] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> SparseMatrix:
    return SparseMatrix(
        torch.ops.dgl_sparse.create_from_coo(row, col, val, shape)
    )

