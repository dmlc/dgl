import pytest
import torch
import sys

import backend as F

from dgl.mock_sparse2 import create_from_coo, create_from_csr, create_from_csc

# FIXME: Skipping tests on win.
if not sys.platform.startswith("linux"):
    pytest.skip("skipping tests on win", allow_module_level=True)


@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("row", [(0, 0, 1, 2), (0, 1, 2, 4)])
@pytest.mark.parametrize("col", [(0, 1, 2, 2), (1, 3, 3, 4)])
@pytest.mark.parametrize("shape", [None, (5, 5), (5, 6)])
def test_create_from_coo(dense_dim, row, col, shape):
    val_shape = (len(row),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    ctx = F.ctx()
    val = torch.randn(val_shape).to(ctx)
    row = torch.tensor(row).to(ctx)
    col = torch.tensor(col).to(ctx)
    mat = create_from_coo(row, col, val, shape)

    if shape is None:
        shape = (torch.max(row).item() + 1, torch.max(col).item() + 1)

    mat_row, mat_col = mat.coo()
    mat_val = mat.val

    assert mat.shape == shape
    assert mat.nnz == row.numel()
    assert mat.dtype == val.dtype
    assert torch.allclose(mat_val, val)
    assert torch.allclose(mat_row, row)
    assert torch.allclose(mat_col, col)


@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("indptr", [(0, 0, 1, 4), (0, 1, 2, 4)])
@pytest.mark.parametrize("indices", [(0, 1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("shape", [None, (3, 5)])
def test_create_from_csr(dense_dim, indptr, indices, shape):
    val_shape = (len(indices),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    ctx = F.ctx()
    val = torch.randn(val_shape).to(ctx)
    indptr = torch.tensor(indptr).to(ctx)
    indices = torch.tensor(indices).to(ctx)
    mat = create_from_csr(indptr, indices, val, shape)

    if shape is None:
        shape = (indptr.numel() - 1, torch.max(indices).item() + 1)

    assert mat.device == val.device
    assert mat.shape == shape
    assert mat.nnz == indices.numel()
    assert mat.dtype == val.dtype
    mat_indptr, mat_indices, value_indices = mat.csr()
    mat_val = mat.val if value_indices is None else mat.val[value_indices]
    assert torch.allclose(mat_indptr, indptr)
    assert torch.allclose(mat_indices, indices)
    assert torch.allclose(mat_val, val)


@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("indptr", [(0, 0, 1, 4), (0, 1, 2, 4)])
@pytest.mark.parametrize("indices", [(0, 1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("shape", [None, (5, 3)])
def test_create_from_csc(dense_dim, indptr, indices, shape):
    val_shape = (len(indices),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    ctx = F.ctx()
    val = torch.randn(val_shape).to(ctx)
    indptr = torch.tensor(indptr).to(ctx)
    indices = torch.tensor(indices).to(ctx)
    mat = create_from_csc(indptr, indices, val, shape)

    if shape is None:
        shape = (torch.max(indices).item() + 1, indptr.numel() - 1)

    assert mat.device == val.device
    assert mat.shape == shape
    assert mat.nnz == indices.numel()
    assert mat.dtype == val.dtype
    mat_indptr, mat_indices, value_indices = mat.csc()
    mat_val = mat.val if value_indices is None else mat.val[value_indices]
    assert torch.allclose(mat_indptr, indptr)
    assert torch.allclose(mat_indices, indices)
    assert torch.allclose(mat_val, val)


@pytest.mark.parametrize("val_shape", [(3), (3, 2)])
def test_dense(val_shape):
    ctx = F.ctx()

    row = torch.tensor([1, 1, 2]).to(ctx)
    col = torch.tensor([2, 4, 3]).to(ctx)
    val = torch.randn(val_shape).to(ctx)
    A = create_from_coo(row, col, val)
    A_dense = A.dense()

    shape = A.shape + val.shape[1:]
    mat = torch.zeros(shape, device=ctx)
    mat[row, col] = val
    assert torch.allclose(A_dense, mat)


def test_set_val():
    ctx = F.ctx()

    row = torch.tensor([1, 1, 2]).to(ctx)
    col = torch.tensor([2, 4, 3]).to(ctx)
    nnz = len(row)
    old_val = torch.ones(nnz).to(ctx)
    A = create_from_coo(row, col, old_val)

    new_val = torch.zeros(nnz).to(ctx)
    A.val = new_val
    assert torch.allclose(new_val, A.val)


@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("indptr", [(0, 0, 1, 4), (0, 1, 2, 4)])
@pytest.mark.parametrize("indices", [(0, 1, 2, 3), (1, 4, 3, 2)])
@pytest.mark.parametrize("shape", [None, (3, 5)])
def test_csr_to_coo(dense_dim, indptr, indices, shape):
    ctx = F.ctx()
    val_shape = (len(indices),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    val = torch.randn(val_shape).to(ctx)
    indptr = torch.tensor(indptr).to(ctx)
    indices = torch.tensor(indices).to(ctx)
    mat = create_from_csr(indptr, indices, val, shape)

    if shape is None:
        shape = (indptr.numel() - 1, torch.max(indices).item() + 1)

    row = (
        torch.arange(0, indptr.shape[0] - 1)
        .to(ctx)
        .repeat_interleave(torch.diff(indptr))
    )
    col = indices
    mat_row, mat_col = mat.coo()
    mat_val = mat.val

    assert mat.shape == shape
    assert mat.nnz == row.numel()
    assert mat.device == row.device
    assert mat.dtype == val.dtype
    assert torch.allclose(mat_val, val)
    assert torch.allclose(mat_row, row)
    assert torch.allclose(mat_col, col)


@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("indptr", [(0, 0, 1, 4), (0, 1, 2, 4)])
@pytest.mark.parametrize("indices", [(0, 1, 2, 3), (1, 4, 3, 2)])
@pytest.mark.parametrize("shape", [None, (5, 3)])
def test_csc_to_coo(dense_dim, indptr, indices, shape):
    ctx = F.ctx()
    val_shape = (len(indices),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    val = torch.randn(val_shape).to(ctx)
    indptr = torch.tensor(indptr).to(ctx)
    indices = torch.tensor(indices).to(ctx)
    mat = create_from_csc(indptr, indices, val, shape)

    if shape is None:
        shape = (torch.max(indices).item() + 1, indptr.numel() - 1)

    col = (
        torch.arange(0, indptr.shape[0] - 1)
        .to(ctx)
        .repeat_interleave(torch.diff(indptr))
    )
    row = indices
    mat_row, mat_col = mat.coo()
    mat_val = mat.val

    assert mat.shape == shape
    assert mat.nnz == row.numel()
    assert mat.device == row.device
    assert mat.dtype == val.dtype
    assert torch.allclose(mat_val, val)
    assert torch.allclose(mat_row, row)
    assert torch.allclose(mat_col, col)


def _scatter_add(a, index, v=1):
    index = index.tolist()
    for i in index:
        a[i] += v
    return a


@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("row", [(0, 0, 1, 2), (0, 1, 2, 4)])
@pytest.mark.parametrize("col", [(0, 1, 2, 2), (1, 3, 3, 4)])
@pytest.mark.parametrize("shape", [None, (5, 5), (5, 6)])
def test_coo_to_csr(dense_dim, row, col, shape):
    val_shape = (len(row),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    ctx = F.ctx()
    val = torch.randn(val_shape).to(ctx)
    row = torch.tensor(row).to(ctx)
    col = torch.tensor(col).to(ctx)
    mat = create_from_coo(row, col, val, shape)

    if shape is None:
        shape = (torch.max(row).item() + 1, torch.max(col).item() + 1)

    mat_indptr, mat_indices, value_indices = mat.csr()
    mat_val = mat.val if value_indices is None else mat.val[value_indices]
    indptr = torch.zeros(shape[0] + 1).to(ctx)
    indptr = _scatter_add(indptr, row + 1)
    indptr = torch.cumsum(indptr, 0).long()
    indices = col

    assert mat.shape == shape
    assert mat.nnz == row.numel()
    assert mat.dtype == val.dtype
    assert torch.allclose(mat_val, val)
    assert torch.allclose(mat_indptr, indptr)
    assert torch.allclose(mat_indices, indices)


@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("indptr", [(0, 0, 1, 4), (0, 1, 2, 4)])
@pytest.mark.parametrize("indices", [(0, 1, 2, 3), (1, 4, 3, 2)])
@pytest.mark.parametrize("shape", [None, (5, 3)])
def test_csc_to_csr(dense_dim, indptr, indices, shape):
    ctx = F.ctx()
    val_shape = (len(indices),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    val = torch.randn(val_shape).to(ctx)
    indptr = torch.tensor(indptr).to(ctx)
    indices = torch.tensor(indices).to(ctx)
    mat = create_from_csc(indptr, indices, val, shape)
    mat_indptr, mat_indices, value_indices = mat.csr()
    mat_val = mat.val if value_indices is None else mat.val[value_indices]

    if shape is None:
        shape = (torch.max(indices).item() + 1, indptr.numel() - 1)

    col = (
        torch.arange(0, indptr.shape[0] - 1)
        .to(ctx)
        .repeat_interleave(torch.diff(indptr))
    )
    row = indices
    row, sort_index = row.sort(stable=True)
    col = col[sort_index]
    val = val[sort_index]
    indptr = torch.zeros(shape[0] + 1).to(ctx)
    indptr = _scatter_add(indptr, row + 1)
    indptr = torch.cumsum(indptr, 0).long()
    indices = col

    assert mat.shape == shape
    assert mat.nnz == row.numel()
    assert mat.device == row.device
    assert mat.dtype == val.dtype
    assert torch.allclose(mat_val, val)
    assert torch.allclose(mat_indptr, indptr)
    assert torch.allclose(mat_indices, indices)


@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("row", [(0, 0, 1, 2), (0, 1, 2, 4)])
@pytest.mark.parametrize("col", [(0, 1, 2, 2), (1, 3, 3, 4)])
@pytest.mark.parametrize("shape", [None, (5, 5), (5, 6)])
def test_coo_to_csc(dense_dim, row, col, shape):

    val_shape = (len(row),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    ctx = F.ctx()
    val = torch.randn(val_shape).to(ctx)
    row = torch.tensor(row).to(ctx)
    col = torch.tensor(col).to(ctx)
    mat = create_from_coo(row, col, val, shape)

    if shape is None:
        shape = (torch.max(row).item() + 1, torch.max(col).item() + 1)

    mat_indptr, mat_indices, value_indices = mat.csc()
    mat_val = mat.val if value_indices is None else mat.val[value_indices]
    indptr = torch.zeros(shape[1] + 1).to(ctx)
    _scatter_add(indptr, col + 1)
    indptr = torch.cumsum(indptr, 0).long()
    indices = row

    assert mat.shape == shape
    assert mat.nnz == row.numel()
    assert mat.dtype == val.dtype
    assert torch.allclose(mat_val, val)
    assert torch.allclose(mat_indptr, indptr)
    assert torch.allclose(mat_indices, indices)


@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("indptr", [(0, 0, 1, 4), (0, 1, 2, 4)])
@pytest.mark.parametrize("indices", [(0, 1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("shape", [None, (3, 5)])
def test_csr_to_csc(dense_dim, indptr, indices, shape):
    val_shape = (len(indices),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    ctx = F.ctx()
    val = torch.randn(val_shape).to(ctx)
    indptr = torch.tensor(indptr).to(ctx)
    indices = torch.tensor(indices).to(ctx)
    mat = create_from_csr(indptr, indices, val, shape)
    mat_indptr, mat_indices, value_indices = mat.csc()
    mat_val = mat.val if value_indices is None else mat.val[value_indices]

    if shape is None:
        shape = (indptr.numel() - 1, torch.max(indices).item() + 1)

    row = (
        torch.arange(0, indptr.shape[0] - 1)
        .to(ctx)
        .repeat_interleave(torch.diff(indptr))
    )

    col = indices
    col, sort_index = col.sort(stable=True)
    row = row[sort_index]
    val = val[sort_index]
    indptr = torch.zeros(shape[1] + 1).to(ctx)
    indptr = _scatter_add(indptr, col + 1)
    indptr = torch.cumsum(indptr, 0).long()
    indices = row

    assert mat.shape == shape
    assert mat.nnz == row.numel()
    assert mat.device == row.device
    assert mat.dtype == val.dtype
    assert torch.allclose(mat_val, val)
    assert torch.allclose(mat_indptr, indptr)
    assert torch.allclose(mat_indices, indices)
