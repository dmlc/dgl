"""Matmul ops for SparseMatrix"""
# pylint: disable=invalid-name
from typing import Union, List
import torch

from .diag_matrix import DiagMatrix, diag
from .sp_matrix import SparseMatrix, create_from_coo

__all__ = [
    'spmm',
    'bspmm'
]

def sparse_dense_mm(A: SparseMatrix, X: torch.Tensor) -> torch.Tensor:
    """Multiply a sparse matrix by a dense matrix

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix of shape (N, M) with values of shape (nnz)
    X : torch.Tensor
        Dense tensor of shape (M, F) or (M)

    Returns
    -------
    torch.Tensor
        The result of multiplication
    """
    return torch.matmul(A.adj, X)

def sparse_sparse_mm(A1: SparseMatrix, A2: SparseMatrix) -> SparseMatrix:
    """Multiply a sparse matrix by a sparse matrix

    Parameters
    ----------
    A1 : SparseMatrix
        Sparse matrix of shape (N, M) with values of shape (nnz1)
    A2 : SparseMatrix
        Sparse matrix of shape (M, P) with values of shape (nnz2)

    Returns
    -------
    SparseMatrix
        The result of multiplication
    """
    result = torch.sparse.mm(A1.adj, A2.adj).coalesce()
    row, col = result.indices()
    return create_from_coo(row=row,
                           col=col,
                           val=result.values(),
                           shape=result.size())

def spmm_sp(A1: SparseMatrix, A2: Union[torch.Tensor, SparseMatrix, DiagMatrix])\
            -> Union[torch.Tensor, SparseMatrix]:
    """Multiply a sparse matrix by a dense/sparse/diagonal matrix

    Parameters
    ----------
    A1 : SparseMatrix
        Matrix of shape (N, M), with values of shape (nnz1)
    A2 : torch.Tensor, SparseMatrix, or DiagMatrix
        Matrix of shape (M, P). If it is a SparseMatrix or DiagMatrix,
        it should have values of shape (nnz2)

    Returns
    -------
    torch.Tensor or SparseMatrix
        The result of multiplication.

        * It is a dense torch tensor if :attr:`A2` is so.
        * It is a SparseMatrix object otherwise.

    Examples
    --------

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([1, 0, 1])
    >>> val = torch.randn(len(row))
    >>> A1 = create_from_coo(row, col, val)
    >>> A2 = torch.randn(2, 3)
    >>> result = A1 @ A2
    """
    assert isinstance(A2, (torch.Tensor, SparseMatrix, DiagMatrix)), \
        f'Expect arg2 to be a torch Tensor, SparseMatrix, or DiagMatrix object, got {type(A2)}'

    val_dim = len(A1.val.shape)
    assert val_dim == 1, f'Expect arg1.val to be a 1D tensor, got {val_dim}D'

    if A1.shape[1] != A2.shape[0]:
        raise ValueError(f'Expect arg1.shape[1] and arg2.shape[0] to be the same, \
                           got {A1.shape[1]} and {A2.shape[0]}')

    if isinstance(A2, torch.Tensor):
        return sparse_dense_mm(A1, A2)

    val_dim = len(A2.val.shape)
    assert val_dim == 1, f'Expect arg2.val to be a 1D tensor, got {val_dim}D'

    if isinstance(A2, DiagMatrix):
        A2 = A2.as_sparse()
    return sparse_sparse_mm(A1, A2)

def diag_diag(A1: DiagMatrix, A2: DiagMatrix) -> DiagMatrix:
    """Multiply a diagonal matrix by a diagonal matrix

    Parameters
    ----------
    A1 : DiagMatrix
        Matrix of shape (N, M), with values of shape (nnz1)
    A2 : DiagMatrix
        Matrix of shape (M, P), with values of shape (nnz2)

    Returns
    -------
    DiagMatrix
        The result of multiplication.
    """
    M, N = A1.shape
    N, P = A2.shape
    common_diag_len = min(M, N, P)
    new_diag_len = min(M, P)
    diag_val = torch.zeros(new_diag_len)
    diag_val[:common_diag_len] = A1.val[:common_diag_len] * A2.val[:common_diag_len]
    return diag(diag_val.to(A1.device), (M, P))

def spmm_diag(A1: DiagMatrix, A2: Union[torch.Tensor, SparseMatrix, DiagMatrix])\
              -> Union[torch.Tensor, SparseMatrix, DiagMatrix]:
    """Multiply a diagonal matrix by a dense/sparse/diagonal matrix

    Parameters
    ----------
    A1 : DiagMatrix
        Matrix of shape (N, M), with values of shape (nnz1)
    A2 : torch.Tensor, SparseMatrix, or DiagMatrix
        Matrix of shape (M, P). If it is a SparseMatrix or DiagMatrix,
        it should have values of shape (nnz2).

    Returns
    -------
    torch.Tensor or SparseMatrix
        The result of multiplication.

        * It is a dense torch tensor if :attr:`A2` is so.
        * It is a DiagMatrix object if :attr:`A2` is so.
        * It is a SparseMatrix object otherwise.

    Examples
    --------

    >>> val = torch.randn(3)
    >>> A1 = diag(val)
    >>> A2 = torch.randn(3, 2)
    >>> result = A1 @ A2
    """
    assert isinstance(A2, (torch.Tensor, SparseMatrix, DiagMatrix)), \
        f'Expect arg2 to be a torch Tensor, SparseMatrix, or DiagMatrix object, got {type(A2)}'

    val_dim = len(A1.val.shape)
    assert val_dim == 1, f'Expect arg1.val to be a 1D tensor, got {val_dim}D'

    if A1.shape[1] != A2.shape[0]:
        raise ValueError(f'Expect arg1.shape[1] and arg2.shape[0] to be the same, \
                           got {A1.shape[1]} and {A2.shape[0]}')

    if isinstance(A2, torch.Tensor):
        return sparse_dense_mm(A1.as_sparse(), A2)

    val_dim = len(A2.val.shape)
    assert val_dim == 1, f'Expect arg2.val to be a 1D tensor, got {val_dim}D'

    if isinstance(A2, DiagMatrix):
        return diag_diag(A1, A2)
    else:
        return sparse_sparse_mm(A1.as_sparse(), A2)

def spmm(A1: Union[SparseMatrix, DiagMatrix], A2: Union[torch.Tensor, SparseMatrix, DiagMatrix])\
         -> Union[torch.Tensor, SparseMatrix, DiagMatrix]:
    """Multiply a sparse/diagonal matrix by a dense/sparse/diagonal matrix

    Parameters
    ----------
    A1 : SparseMatrix or DiagMatrix
        Matrix of shape (N, M), with values of shape (nnz1)
    A2 : torch.Tensor, SparseMatrix, or DiagMatrix
        Matrix of shape (M, P). If it is a SparseMatrix or DiagMatrix,
        it should have values of shape (nnz2)

    Returns
    -------
    torch.Tensor or SparseMatrix or DiagMatrix
        The result of multiplication.

        * It is a dense torch tensor if :attr:`A2` is so.
        * It is a DiagMatrix object if both :attr:`A1` and :attr:`A2` are so.
        * It is a SparseMatrix object otherwise.

    Examples
    --------

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([1, 0, 1])
    >>> val = torch.randn(len(row))
    >>> A1 = create_from_coo(row, col, val)
    >>> A2 = torch.randn(2, 3)
    >>> result = A1 @ A2
    """
    if isinstance(A1, SparseMatrix):
        return spmm_sp(A1, A2)
    elif isinstance(A1, DiagMatrix):
        return spmm_diag(A1, A2)
    else:
        raise ValueError(f'Expect A1 to be a SparseMatrix or DiagMatrix, got {type(A1)}')

def unbatch_tensor(A: Union[torch.Tensor, SparseMatrix, DiagMatrix])\
                   -> Union[List[torch.Tensor], List[SparseMatrix], List[DiagMatrix]]:
    """Unbatch a tensor, sparse matrix, or diagonal matrix

    Parameters
    ----------
    A : torch.Tensor or SparseMatrix, or DiagMatrix
        Batched matrix/tensor

    Returns
    -------
    list[torch.Tensor] or list[SparseMatrix] or list[DiagMatrix]
        Unbatched matrices/tensors
    """
    if isinstance(A, torch.Tensor):
        return [A[..., i] for i in range(A.shape[-1])]
    elif isinstance(A, SparseMatrix):
        return [
            create_from_coo(row=A.row, col=A.col, val=A.val[:, i], shape=A.shape)
            for i in range(A.val.shape[-1])]
    else:
        return [diag(A.val[:, i], A.shape) for i in range(A.val.shape[-1])]

def batch_tensor(A_list: Union[List[torch.Tensor], List[SparseMatrix], List[DiagMatrix]])\
                 -> Union[torch.Tensor, SparseMatrix, DiagMatrix]:
    """Batch a list of tensors, sparse matrices, or diagonal matrices

    Parameters
    ----------
    A_list : list[torch.Tensor] or list[SparseMatrix] or list[DiagMatrix]
        A list of tensors, sparse matrices, or diagonal matrices

    Returns
    -------
    torch.Tensor or SparseMatrix, or DiagMatrix
        Batched matrix/tensor
    """
    A = A_list[0]
    if isinstance(A, torch.Tensor):
        return torch.stack(A_list, dim=-1)
    elif isinstance(A, SparseMatrix):
        return create_from_coo(
            row=A.row, col=A.col,
            val=torch.stack([A_list[i].val for i in range(len(A_list))], dim=-1), shape=A.shape)
    else:
        return diag(
            val=torch.stack([A_list[i].val for i in range(len(A_list))], dim=-1), shape=A.shape)

def bspmm(A1: Union[SparseMatrix, DiagMatrix], A2: Union[torch.Tensor, SparseMatrix, DiagMatrix])\
          -> Union[torch.Tensor, SparseMatrix, DiagMatrix]:
    """Batched multiplication of a sparse/diagonal matrix by a dense/sparse/diagonal matrix,
    with the last dimension being the batch dimension

    We may consider a SparseMatrix/DiagMatrix with shape (N, M) and values of shape (nnz1, H)
    to be a tensor of shape (N, M, H). The result is then obtained by

    .. code::

        result = []
        for i in range(H):
            # If A2 is a 2D torch Tensor, then this will be A2[:, i]
            result.append(A1[:, :, i] @ A2[:, :, i])
        result = torch.stack(result, dim=-1)

    Parameters
    ----------
    A1 : SparseMatrix or DiagMatrix
        Matrix of shape (N, M), with values of shape (nnz1, H)
    A2 : torch.Tensor, SparseMatrix, or DiagMatrix
        Matrix of shape (M, P). If it is a SparseMatrix or DiagMatrix,
        it should have values of shape (nnz2, H).

    Returns
    -------
    torch.Tensor or SparseMatrix or DiagMatrix
        The result of multiplication.

        * It is a dense torch tensor if :attr:`A2` is so.
        * It is a DiagMatrix object if both :attr:`A1` and :attr:`A2` are so.
        * It is a SparseMatrix object otherwise.

    Examples
    --------

    >>> row = torch.tensor([0, 1, 1])
    >>> col = torch.tensor([1, 0, 1])
    >>> H = 4
    >>> val = torch.randn(len(row), H)
    >>> A1 = create_from_coo(row, col, val)
    >>> A2 = torch.randn(2, 3, H)
    >>> result = A1 @ A2
    >>> result.shape
    (2, 3, 4)
    """
    assert isinstance(A1, (SparseMatrix, DiagMatrix)), \
        f'Expect A1 to be a SparseMatrix or DiagMatrix object, got {type(A1)}'
    assert isinstance(A2, (torch.Tensor, SparseMatrix, DiagMatrix)), \
        f'Expect A2 to be a torch Tensor, SparseMatrix, or DiagMatrix object, got {type(A2)}'

    val_dim = len(A1.val.shape)
    assert val_dim == 2, f'Expect A1.val to be a 2D tensor, got {val_dim}D'
    H1 = A1.val.shape[-1]
    if isinstance(A2, torch.Tensor):
        val_dim = len(A2.shape)
        assert val_dim in [2, 3], f'Expect to be a 2D/3D tensor, got {val_dim}D'
        H2 = A2.shape[-1]
        assert H1 == H2, f'Expect A1.val.shape[-1] == A2.shape[-1], got {H1} and {H2}'
    else:
        val_dim = len(A2.val.shape)
        assert val_dim == 2, f'Expect A2.val to be a 2D tensor, got {val_dim}D'
        H2 = A2.val.shape[-1]
        assert H1 == H2, f'Expect A1.val.shape[-1] == A2.val.shape[-1], got {H1} and {H2}'

    A1_unbatched = unbatch_tensor(A1)
    A2_unbatched = unbatch_tensor(A2)
    results = [spmm(A1_unbatched[i], A2_unbatched[i]) for i in range(H1)]
    return batch_tensor(results)

SparseMatrix.__matmul__ = spmm_sp
DiagMatrix.__matmul__ = spmm_diag
