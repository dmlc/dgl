import torch
from dgl.mock_sparse2 import (
    create_from_coo,
    create_from_csc,
    create_from_csr,
    SparseMatrix,
)


def clone_detach_and_grad(t):
    t = t.clone().detach()
    t.requires_grad_()
    return t


def rand_coo(shape, nnz, dev):
    row = torch.randint(0, shape[0], (nnz,), device=dev)
    col = torch.randint(0, shape[1], (nnz,), device=dev)
    val = torch.randn(nnz, device=dev, requires_grad=True)
    return create_from_coo(row, col, val, shape)


def rand_csr(shape, nnz, dev):
    row = torch.randint(0, shape[0], (nnz,), device=dev)
    col = torch.randint(0, shape[1], (nnz,), device=dev)
    val = torch.randn(nnz, device=dev, requires_grad=True)
    indptr = torch.zeros(shape[0] + 1, device=dev, dtype=torch.int64)
    for r in row.tolist():
        indptr[r + 1] += 1
    indptr = torch.cumsum(indptr, 0)
    indices = col
    return create_from_csr(indptr, indices, val, shape=shape)


def rand_csc(shape, nnz, dev):
    row = torch.randint(0, shape[0], (nnz,), device=dev)
    col = torch.randint(0, shape[1], (nnz,), device=dev)
    val = torch.randn(nnz, device=dev, requires_grad=True)
    indptr = torch.zeros(shape[1] + 1, device=dev, dtype=torch.int64)
    for c in col.tolist():
        indptr[c + 1] += 1
    indptr = torch.cumsum(indptr, 0)
    indices = row
    return create_from_csc(indptr, indices, val, shape=shape)


def sparse_matrix_to_dense(A: SparseMatrix):
    dense = A.dense()
    dense.requires_grad_()
    return dense


def sparse_matrix_to_torch_sparse(A: SparseMatrix):
    row, col = A.coo()
    edge_index = torch.cat((row.unsqueeze(0), col.unsqueeze(0)), 0)
    shape = A.shape
    val = A.val.clone().detach()
    if len(A.val.shape) > 1:
        shape += (A.val.shape[-1],)
    ret = torch.sparse_coo_tensor(edge_index, val, shape).coalesce()
    ret.requires_grad_()
    return ret
