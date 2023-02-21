import numpy as np
import torch

from dgl.sparse import from_coo, from_csc, from_csr, SparseMatrix

np.random.seed(42)
torch.random.manual_seed(42)


def clone_detach_and_grad(t):
    t = t.clone().detach()
    t.requires_grad_()
    return t


def rand_coo(shape, nnz, dev, nz_dim=None):
    # Create a sparse matrix without duplicate entries.
    nnzid = np.random.choice(shape[0] * shape[1], nnz, replace=False)
    nnzid = torch.tensor(nnzid, device=dev).long()
    row = torch.div(nnzid, shape[1], rounding_mode="floor")
    col = nnzid % shape[1]
    if nz_dim is None:
        val = torch.randn(nnz, device=dev, requires_grad=True)
    else:
        val = torch.randn(nnz, nz_dim, device=dev, requires_grad=True)
    return from_coo(row, col, val, shape)


def rand_csr(shape, nnz, dev, nz_dim=None):
    # Create a sparse matrix without duplicate entries.
    nnzid = np.random.choice(shape[0] * shape[1], nnz, replace=False)
    nnzid = torch.tensor(nnzid, device=dev).long()
    row = torch.div(nnzid, shape[1], rounding_mode="floor")
    col = nnzid % shape[1]
    if nz_dim is None:
        val = torch.randn(nnz, device=dev, requires_grad=True)
    else:
        val = torch.randn(nnz, nz_dim, device=dev, requires_grad=True)
    indptr = torch.zeros(shape[0] + 1, device=dev, dtype=torch.int64)
    for r in row.tolist():
        indptr[r + 1] += 1
    indptr = torch.cumsum(indptr, 0)
    row_sorted, row_sorted_idx = torch.sort(row)
    indices = col[row_sorted_idx]
    return from_csr(indptr, indices, val, shape=shape)


def rand_csc(shape, nnz, dev, nz_dim=None):
    # Create a sparse matrix without duplicate entries.
    nnzid = np.random.choice(shape[0] * shape[1], nnz, replace=False)
    nnzid = torch.tensor(nnzid, device=dev).long()
    row = torch.div(nnzid, shape[1], rounding_mode="floor")
    col = nnzid % shape[1]
    if nz_dim is None:
        val = torch.randn(nnz, device=dev, requires_grad=True)
    else:
        val = torch.randn(nnz, nz_dim, device=dev, requires_grad=True)
    indptr = torch.zeros(shape[1] + 1, device=dev, dtype=torch.int64)
    for c in col.tolist():
        indptr[c + 1] += 1
    indptr = torch.cumsum(indptr, 0)
    col_sorted, col_sorted_idx = torch.sort(col)
    indices = row[col_sorted_idx]
    return from_csc(indptr, indices, val, shape=shape)


def rand_coo_uncoalesced(shape, nnz, dev):
    # Create a sparse matrix with possible duplicate entries.
    row = torch.randint(shape[0], (nnz,), device=dev)
    col = torch.randint(shape[1], (nnz,), device=dev)
    val = torch.randn(nnz, device=dev, requires_grad=True)
    return from_coo(row, col, val, shape)


def rand_csr_uncoalesced(shape, nnz, dev):
    # Create a sparse matrix with possible duplicate entries.
    row = torch.randint(shape[0], (nnz,), device=dev)
    col = torch.randint(shape[1], (nnz,), device=dev)
    val = torch.randn(nnz, device=dev, requires_grad=True)
    indptr = torch.zeros(shape[0] + 1, device=dev, dtype=torch.int64)
    for r in row.tolist():
        indptr[r + 1] += 1
    indptr = torch.cumsum(indptr, 0)
    row_sorted, row_sorted_idx = torch.sort(row)
    indices = col[row_sorted_idx]
    return from_csr(indptr, indices, val, shape=shape)


def rand_csc_uncoalesced(shape, nnz, dev):
    # Create a sparse matrix with possible duplicate entries.
    row = torch.randint(shape[0], (nnz,), device=dev)
    col = torch.randint(shape[1], (nnz,), device=dev)
    val = torch.randn(nnz, device=dev, requires_grad=True)
    indptr = torch.zeros(shape[1] + 1, device=dev, dtype=torch.int64)
    for c in col.tolist():
        indptr[c + 1] += 1
    indptr = torch.cumsum(indptr, 0)
    col_sorted, col_sorted_idx = torch.sort(col)
    indices = row[col_sorted_idx]
    return from_csc(indptr, indices, val, shape=shape)


def sparse_matrix_to_dense(A: SparseMatrix):
    dense = A.to_dense()
    return clone_detach_and_grad(dense)


def sparse_matrix_to_torch_sparse(A: SparseMatrix, val=None):
    row, col = A.coo()
    edge_index = torch.cat((row.unsqueeze(0), col.unsqueeze(0)), 0)
    shape = A.shape
    if val is None:
        val = A.val
    val = val.clone().detach()
    if len(A.val.shape) > 1:
        shape += (A.val.shape[-1],)
    ret = torch.sparse_coo_tensor(edge_index, val, shape).coalesce()
    ret.requires_grad_()
    return ret


def dense_mask(dense, sparse):
    ret = torch.zeros_like(dense)
    row, col = sparse.coo()
    for r, c in zip(row, col):
        ret[r, c] = dense[r, c]
    return ret
