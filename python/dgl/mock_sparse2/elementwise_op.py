"""DGL elementwise operator module."""
import torch

from .sparse_matrix import SparseMatrix


def sp_add(A: SparseMatrix, B: SparseMatrix) -> SparseMatrix:
    """TODO Add op for DiagMatrix"""
    return SparseMatrix(
        torch.ops.dgl_sparse.spspadd(A.c_sparse_matrix, B.c_sparse_matrix)
    )


SparseMatrix.__add__ = sp_add
