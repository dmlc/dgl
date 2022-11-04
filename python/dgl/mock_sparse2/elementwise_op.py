"""DGL elementwise operator module."""
from typing import Union

from .diag_matrix import DiagMatrix
from .elementwise_op_diag import diag_add
from .elementwise_op_sp import sp_add
from .sparse_matrix import SparseMatrix

__all__ = ["add"]


def add(
    A: Union[SparseMatrix, DiagMatrix], B: Union[SparseMatrix, DiagMatrix]
) -> Union[SparseMatrix, DiagMatrix]:
    """Elementwise addition"""
    if isinstance(A, DiagMatrix) and isinstance(B, DiagMatrix):
        return diag_add(A, B)
    return sp_add(A, B)
