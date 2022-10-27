""" DGL sparse library C++ binding"""
import os
import torch
from typing import List, Optional, Tuple

package_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
so_path = os.path.join(
    package_path, "../tensoradapter/pytorch/libdgl_sparse.so"
)
torch.classes.load_library(so_path)

_C_SparseMatrix = torch.classes.dgl_sparse.SparseMatrix


class SparseMatrix:
    def __init__(self, c_sparse_matrix: _C_SparseMatrix):
        self._c_sparse_matrix = c_sparse_matrix

    @property
    def val(self) -> torch.Tensor:
        """Get the values of the nonzero elements.

        Returns
        -------
        tensor
            Values of the nonzero elements
        """
        return self._c_sparse_matrix.val()

    @property
    def shape(self) -> List[int]:
        """Shape of the sparse matrix.

        Returns
        -------
        List[int]
            The shape of the matrix
        """
        return self._c_sparse_matrix.shape()


def create_from_coo(
    row: torch.Tensor,
    col: torch.Tensor,
    val: Optional[torch.Tensor] = None,
    shape: Optional[Tuple(int, int)] = None,
) -> SparseMatrix:
    return SparseMatrix(
        torch.ops.dgl_sparse.create_from_coo(row, col, val, shape)
    )
