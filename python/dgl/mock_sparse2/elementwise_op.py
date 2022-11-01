"""DGL elementwise operator module."""
from .sparse_matrix import SparseMatrix
import torch

# TODO: Add addition for DiagMatrix
def sp_add(A: SparseMatrix, B: SparseMatrix) -> SparseMatrix:
    return SparseMatrix(
        torch.ops.dgl_sparse.spspadd(
            A.c_sparse_matrix, B.c_sparse_matrix
        )
    )


SparseMatrix.__add__ = sp_add
