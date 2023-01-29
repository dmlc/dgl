import sys
import unittest

import backend as F
import pytest
import torch

from dgl.sparse import from_coo, from_csc, from_csr, val_like

# TODO(#4818): Skipping tests on win.
if not sys.platform.startswith("linux"):
    pytest.skip("skipping tests on win", allow_module_level=True)


@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("row", [(0, 0, 1, 2), (0, 1, 2, 4)])
@pytest.mark.parametrize("col", [(0, 1, 2, 2), (1, 3, 3, 4)])
@pytest.mark.parametrize("shape", [None, (5, 5), (5, 6)])
def test_from_coo(dense_dim, row, col, shape):
    val_shape = (len(row),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    ctx = F.ctx()
    val = torch.randn(val_shape).to(ctx)
    row = torch.tensor(row).to(ctx)
    col = torch.tensor(col).to(ctx)
    mat = from_coo(row, col, val, shape)

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
def test_from_csr(dense_dim, indptr, indices, shape):
    val_shape = (len(indices),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    ctx = F.ctx()
    val = torch.randn(val_shape).to(ctx)
    indptr = torch.tensor(indptr).to(ctx)
    indices = torch.tensor(indices).to(ctx)
    mat = from_csr(indptr, indices, val, shape)

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
def test_from_csc(dense_dim, indptr, indices, shape):
    val_shape = (len(indices),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    ctx = F.ctx()
    val = torch.randn(val_shape).to(ctx)
    indptr = torch.tensor(indptr).to(ctx)
    indices = torch.tensor(indices).to(ctx)
    mat = from_csc(indptr, indices, val, shape)

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
    A = from_coo(row, col, val)
    A_dense = A.to_dense()

    shape = A.shape + val.shape[1:]
    mat = torch.zeros(shape, device=ctx)
    mat[row, col] = val
    assert torch.allclose(A_dense, mat)


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
    mat = from_csr(indptr, indices, val, shape)

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
    mat = from_csc(indptr, indices, val, shape)

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
    mat = from_coo(row, col, val, shape)

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
    mat = from_csc(indptr, indices, val, shape)
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
    mat = from_coo(row, col, val, shape)

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
    mat = from_csr(indptr, indices, val, shape)
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


@pytest.mark.parametrize("val_shape", [(3), (3, 2)])
@pytest.mark.parametrize("shape", [(3, 5), (5, 5)])
def test_val_like(val_shape, shape):
    def check_val_like(A, B):
        assert A.shape == B.shape
        assert A.nnz == B.nnz
        assert torch.allclose(torch.stack(A.coo()), torch.stack(B.coo()))
        assert A.val.device == B.val.device

    ctx = F.ctx()

    # COO
    row = torch.tensor([1, 1, 2]).to(ctx)
    col = torch.tensor([2, 4, 3]).to(ctx)
    val = torch.randn(3).to(ctx)
    coo_A = from_coo(row, col, val, shape)
    new_val = torch.randn(val_shape).to(ctx)
    coo_B = val_like(coo_A, new_val)
    check_val_like(coo_A, coo_B)

    # CSR
    indptr, indices, _ = coo_A.csr()
    csr_A = from_csr(indptr, indices, val, shape)
    csr_B = val_like(csr_A, new_val)
    check_val_like(csr_A, csr_B)

    # CSC
    indptr, indices, _ = coo_A.csc()
    csc_A = from_csc(indptr, indices, val, shape)
    csc_B = val_like(csc_A, new_val)
    check_val_like(csc_A, csc_B)


def test_coalesce():
    ctx = F.ctx()

    row = torch.tensor([1, 0, 0, 0, 1]).to(ctx)
    col = torch.tensor([1, 1, 1, 2, 2]).to(ctx)
    val = torch.arange(len(row)).to(ctx)
    A = from_coo(row, col, val, (4, 4))

    assert A.has_duplicate()

    A_coalesced = A.coalesce()

    assert A_coalesced.nnz == 4
    assert A_coalesced.shape == (4, 4)
    assert list(A_coalesced.row) == [0, 0, 1, 1]
    assert list(A_coalesced.col) == [1, 2, 1, 2]
    # Values of duplicate indices are added together.
    assert list(A_coalesced.val) == [3, 3, 0, 4]
    assert not A_coalesced.has_duplicate()


def test_has_duplicate():
    ctx = F.ctx()

    row = torch.tensor([1, 0, 0, 0, 1]).to(ctx)
    col = torch.tensor([1, 1, 1, 2, 2]).to(ctx)
    val = torch.arange(len(row)).to(ctx)
    shape = (4, 4)

    # COO
    coo_A = from_coo(row, col, val, shape)
    assert coo_A.has_duplicate()

    # CSR
    indptr, indices, _ = coo_A.csr()
    csr_A = from_csr(indptr, indices, val, shape)
    assert csr_A.has_duplicate()

    # CSC
    indptr, indices, _ = coo_A.csc()
    csc_A = from_csc(indptr, indices, val, shape)
    assert csc_A.has_duplicate()


def test_print():
    ctx = F.ctx()

    # basic
    row = torch.tensor([1, 1, 3]).to(ctx)
    col = torch.tensor([2, 1, 3]).to(ctx)
    val = torch.tensor([1.0, 1.0, 2.0]).to(ctx)
    A = from_coo(row, col, val)
    print(A)

    # vector-shape non zero
    row = torch.tensor([1, 1, 3]).to(ctx)
    col = torch.tensor([2, 1, 3]).to(ctx)
    val = torch.randn(3, 2).to(ctx)
    A = from_coo(row, col, val)
    print(A)


@unittest.skipIf(
    F._default_context_str == "cpu",
    reason="Device conversions don't need to be tested on CPU.",
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_to_device(device):
    row = torch.tensor([1, 1, 2])
    col = torch.tensor([1, 2, 0])
    mat = from_coo(row, col, shape=(3, 4))

    target_row = row.to(device)
    target_col = col.to(device)
    target_val = mat.val.to(device)

    mat2 = mat.to(device=device)
    assert mat2.shape == mat.shape
    assert torch.allclose(mat2.row, target_row)
    assert torch.allclose(mat2.col, target_col)
    assert torch.allclose(mat2.val, target_val)

    mat2 = getattr(mat, device)()
    assert mat2.shape == mat.shape
    assert torch.allclose(mat2.row, target_row)
    assert torch.allclose(mat2.col, target_col)
    assert torch.allclose(mat2.val, target_val)


@pytest.mark.parametrize(
    "dtype", [torch.float, torch.double, torch.int, torch.long]
)
def test_to_dtype(dtype):
    row = torch.tensor([1, 1, 2])
    col = torch.tensor([1, 2, 0])
    mat = from_coo(row, col, shape=(3, 4))

    target_val = mat.val.to(dtype=dtype)

    mat2 = mat.to(dtype=dtype)
    assert mat2.shape == mat.shape
    assert torch.allclose(mat2.val, target_val)

    func_name = {
        torch.float: "float",
        torch.double: "double",
        torch.int: "int",
        torch.long: "long",
    }
    mat2 = getattr(mat, func_name[dtype])()
    assert mat2.shape == mat.shape
    assert torch.allclose(mat2.val, target_val)
