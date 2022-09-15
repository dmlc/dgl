"""dgl sddmm operators for sparse matrix module."""
import torch
from sp_matrix import SparseMatrix

__all__ = ['sddmm']

def sddmm(A: SparseMatrix, matB: torch.tensor, matC: torch.tensor) -> SparseMatrix:
    r""" Sampled-Dense-Dense Matrix Multiplication.

    .. math::
        out = (matB @ matC) * A

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix of shape (M, N)
    matB : Tensor
        Dense matrix of shape (M, K)
    matC : Tensor
        Dense matrix of shape (K, N)

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    # TODO(Israt): Double check the example in the console
    Case 1: Add two matrices of same sparsity structure

    >>> row = torch.tensor([1, 1, 2])
    >>> col = torch.tensor([2, 3, 3])
    >>> val = torch.tensor([1.1, 1, 2])
    >>> A = SparseMatrix(src, dst, val, (4,5))
    >>> mat1 = torch.randn(4, 3)
    >>> mat2 = torch.randn(3, 5)
    >>> sddmm(A, mat1, mat2)
    SparseMatrix(indices=tensor([[1, 1, 2],
            [2, 3, 3]]),
    values=tensor([ 1.7648, -1.4142, -1.0356]),
    shape=(4, 5), nnz=3)
    """
    # PyTorch's sddmm operator only supports CSR format
    A = A.adj.to_sparse_csr()
    print(A.shape, matB.shape, matC.shape)
    Out = (torch.sparse.sampled_addmm(A, matB, matC)).to_sparse_coo()
    return SparseMatrix(Out.indices()[0], Out.indices()[1], Out.values(), Out.shape)
