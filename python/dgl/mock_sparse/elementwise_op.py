"""DGL elementwise operator module."""
from typing import Union

from .diag_matrix import DiagMatrix
from .elementwise_op_diag import *
from .elementwise_op_sp import *
from .sp_matrix import SparseMatrix

__all__ = ["add", "sub", "mul", "div", "power"]


def add(
    A: Union[SparseMatrix, DiagMatrix], B: Union[SparseMatrix, DiagMatrix]
) -> Union[SparseMatrix, DiagMatrix]:
    """Elementwise addition"""
    if isinstance(A, DiagMatrix) and isinstance(B, DiagMatrix):
        return diag_add(A, B)
    return sp_add(A, B)


def sub(
    A: Union[SparseMatrix, DiagMatrix], B: Union[SparseMatrix, DiagMatrix]
) -> Union[SparseMatrix, DiagMatrix]:
    """Elementwise addition"""
    if isinstance(A, DiagMatrix) and isinstance(B, DiagMatrix):
        return diag_sub(A, B)
    return sp_sub(A, B)


def mul(
    A: Union[SparseMatrix, DiagMatrix, float],
    B: Union[SparseMatrix, DiagMatrix, float],
) -> Union[SparseMatrix, DiagMatrix]:
    """Elementwise multiplication"""
    if isinstance(A, SparseMatrix) or isinstance(B, SparseMatrix):
        return sp_mul(A, B)
    return diag_mul(A, B)


def div(
    A: Union[SparseMatrix, DiagMatrix],
    B: Union[SparseMatrix, DiagMatrix, float],
) -> Union[SparseMatrix, DiagMatrix]:
    """Elementwise division"""
    if isinstance(A, SparseMatrix) or isinstance(B, SparseMatrix):
        return sp_div(A, B)
    return diag_div(A, B)


def power(A: SparseMatrix, B: float) -> Union[SparseMatrix, DiagMatrix]:
    if isinstance(A, SparseMatrix) or isinstance(B, SparseMatrix):
        return sp_power(A, B)
    return diag_power(A, B)
