import unittest
import warnings

import backend as F
import pytest
import torch

from dgl.sparse import (
    diag,
    from_coo,
    from_csc,
    from_csr,
    from_torch_sparse,
    identity,
    to_torch_sparse_coo,
    to_torch_sparse_csc,
    to_torch_sparse_csr,
    val_like,
)

from .utils import (
    rand_coo,
    rand_csc,
    rand_csr,
    rand_diag,
    sparse_matrix_to_dense,
)


def _torch_sparse_csr_tensor(indptr, indices, val, torch_sparse_shape):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return torch.sparse_csr_tensor(indptr, indices, val, torch_sparse_shape)


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


@pytest.mark.parametrize("shape", [(3, 5), (5, 5), (5, 4)])
def test_diag_conversions(shape):
    n_rows, n_cols = shape
    nnz = min(shape)
    ctx = F.ctx()
    val = torch.randn(nnz).to(ctx)
    D = diag(val, shape)
    row, col = D.coo()
    assert torch.allclose(row, torch.arange(nnz).to(ctx))
    assert torch.allclose(col, torch.arange(nnz).to(ctx))

    indptr, indices, _ = D.csr()
    exp_indptr = list(range(0, nnz + 1)) + [nnz] * (n_rows - nnz)
    assert torch.allclose(indptr, torch.tensor(exp_indptr).to(ctx))
    assert torch.allclose(indices, torch.arange(nnz).to(ctx))

    indptr, indices, _ = D.csc()
    exp_indptr = list(range(0, nnz + 1)) + [nnz] * (n_cols - nnz)
    assert torch.allclose(indptr, torch.tensor(exp_indptr).to(ctx))
    assert torch.allclose(indices, torch.arange(nnz).to(ctx))


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


@pytest.mark.parametrize(
    "create_func", [rand_diag, rand_csr, rand_csc, rand_coo]
)
@pytest.mark.parametrize("shape", [(5, 5), (6, 4)])
@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("select_dim", [0, 1])
@pytest.mark.parametrize("index", [(0, 1, 3), (1, 2)])
def test_index_select(create_func, shape, dense_dim, select_dim, index):
    ctx = F.ctx()
    A = create_func(shape, 20, ctx, dense_dim)
    index = torch.tensor(index).to(ctx)
    A_select = A.index_select(select_dim, index)

    dense = sparse_matrix_to_dense(A)
    dense_select = torch.index_select(dense, select_dim, index)

    A_select_to_dense = sparse_matrix_to_dense(A_select)

    assert A_select_to_dense.shape == dense_select.shape
    assert torch.allclose(A_select_to_dense, dense_select)


@pytest.mark.parametrize(
    "create_func", [rand_diag, rand_csr, rand_csc, rand_coo]
)
@pytest.mark.parametrize("shape", [(5, 5), (6, 4)])
@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("select_dim", [0, 1])
@pytest.mark.parametrize("rang", [slice(0, 2), slice(1, 3)])
def test_range_select(create_func, shape, dense_dim, select_dim, rang):
    ctx = F.ctx()
    A = create_func(shape, 20, ctx, dense_dim)
    A_select = A.range_select(select_dim, rang)

    dense = sparse_matrix_to_dense(A)
    if select_dim == 0:
        dense_select = dense[rang, :]
    else:
        dense_select = dense[:, rang]

    A_select_to_dense = sparse_matrix_to_dense(A_select)

    assert A_select_to_dense.shape == dense_select.shape
    assert torch.allclose(A_select_to_dense, dense_select)


@pytest.mark.parametrize(
    "create_func", [rand_diag, rand_csr, rand_csc, rand_coo]
)
@pytest.mark.parametrize("index", [(0, 1, 2, 3, 4), (0, 1, 3), (1, 1, 2)])
@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_sample_rowwise(create_func, index, replace, bias):
    ctx = F.ctx()
    shape = (5, 5)
    sample_dim = 0
    sample_num = 3
    A = create_func(shape, 10, ctx)
    A = val_like(A, torch.abs(A.val))

    index = torch.tensor(index).to(ctx)

    A_sample = A.sample(sample_dim, sample_num, index, replace, bias)
    A_dense = sparse_matrix_to_dense(A)
    A_sample_to_dense = sparse_matrix_to_dense(A_sample)

    ans_shape = (index.size(0), shape[1])
    # Verify sample elements in origin rows
    for i, row in enumerate(list(index)):
        ans_ele = list(A_dense[row, :].nonzero().reshape(-1))
        ret_ele = list(A_sample_to_dense[i, :].nonzero().reshape(-1))
        for e in ret_ele:
            assert e in ans_ele
        if replace:
            # The number of sample elements in one row should be equal to
            # 'sample_num' if the row is not empty otherwise should be
            # equal to 0.
            assert list(A_sample.row).count(torch.tensor(i)) == (
                sample_num if len(ans_ele) != 0 else 0
            )
        else:
            assert len(ret_ele) == min(sample_num, len(ans_ele))

    assert A_sample.shape == ans_shape
    if not replace:
        assert not A_sample.has_duplicate()


@pytest.mark.parametrize(
    "create_func", [rand_diag, rand_csr, rand_csc, rand_coo]
)
@pytest.mark.parametrize("index", [(0, 1, 2, 3, 4), (0, 1, 3), (1, 1, 2)])
@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_sample_columnwise(create_func, index, replace, bias):
    ctx = F.ctx()
    shape = (5, 5)
    sample_dim = 1
    sample_num = 3
    A = create_func(shape, 10, ctx)
    A = val_like(A, torch.abs(A.val))

    index = torch.tensor(index).to(ctx)

    A_sample = A.sample(sample_dim, sample_num, index, replace, bias)
    A_dense = sparse_matrix_to_dense(A)
    A_sample_to_dense = sparse_matrix_to_dense(A_sample)

    ans_shape = (shape[0], index.size(0))
    # Verify sample elements in origin columns
    for i, col in enumerate(list(index)):
        ans_ele = list(A_dense[:, col].nonzero().reshape(-1))
        ret_ele = list(A_sample_to_dense[:, i].nonzero().reshape(-1))
        for e in ret_ele:
            assert e in ans_ele
        if replace:
            # The number of sample elements in one column should be equal to
            # 'sample_num' if the column is not empty otherwise should be
            # equal to 0.
            assert list(A_sample.col).count(torch.tensor(i)) == (
                sample_num if len(ans_ele) != 0 else 0
            )
        else:
            assert len(ret_ele) == min(sample_num, len(ans_ele))

    assert A_sample.shape == ans_shape
    if not replace:
        assert not A_sample.has_duplicate()


def test_print():
    ctx = F.ctx()

    # basic
    row = torch.tensor([1, 1, 3]).to(ctx)
    col = torch.tensor([2, 1, 3]).to(ctx)
    val = torch.tensor([1.0, 1.0, 2.0]).to(ctx)
    A = from_coo(row, col, val)
    expected = (
        str(
            """SparseMatrix(indices=tensor([[1, 1, 3],
                             [2, 1, 3]]),
             values=tensor([1., 1., 2.]),
             shape=(4, 4), nnz=3)"""
        )
        if str(ctx) == "cpu"
        else str(
            """SparseMatrix(indices=tensor([[1, 1, 3],
                             [2, 1, 3]], device='cuda:0'),
             values=tensor([1., 1., 2.], device='cuda:0'),
             shape=(4, 4), nnz=3)"""
        )
    )
    assert str(A) == expected, print(A, expected)

    # vector-shape non zero
    row = torch.tensor([1, 1, 3]).to(ctx)
    col = torch.tensor([2, 1, 3]).to(ctx)
    val = torch.tensor(
        [[1.3080, 1.5984], [-0.4126, 0.7250], [-0.5416, -0.7022]]
    ).to(ctx)
    A = from_coo(row, col, val)
    expected = (
        str(
            """SparseMatrix(indices=tensor([[1, 1, 3],
                             [2, 1, 3]]),
             values=tensor([[ 1.3080,  1.5984],
                            [-0.4126,  0.7250],
                            [-0.5416, -0.7022]]),
             shape=(4, 4), nnz=3, val_size=(2,))"""
        )
        if str(ctx) == "cpu"
        else str(
            """SparseMatrix(indices=tensor([[1, 1, 3],
                             [2, 1, 3]], device='cuda:0'),
             values=tensor([[ 1.3080,  1.5984],
                            [-0.4126,  0.7250],
                            [-0.5416, -0.7022]], device='cuda:0'),
             shape=(4, 4), nnz=3, val_size=(2,))"""
        )
    )
    assert str(A) == expected, print(A, expected)


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


@pytest.mark.parametrize("dense_dim", [None, 2])
@pytest.mark.parametrize("row", [[0, 0, 1, 2], (0, 1, 2, 4)])
@pytest.mark.parametrize("col", [(0, 1, 2, 2), (1, 3, 3, 4)])
@pytest.mark.parametrize("extra_shape", [(0, 1), (2, 1)])
def test_sparse_matrix_transpose(dense_dim, row, col, extra_shape):
    mat_shape = (max(row) + 1 + extra_shape[0], max(col) + 1 + extra_shape[1])
    val_shape = (len(row),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    ctx = F.ctx()
    val = torch.randn(val_shape).to(ctx)
    row = torch.tensor(row).to(ctx)
    col = torch.tensor(col).to(ctx)
    mat = from_coo(row, col, val, mat_shape).transpose()
    mat_row, mat_col = mat.coo()
    mat_val = mat.val

    assert mat.shape == mat_shape[::-1]
    assert torch.allclose(mat_val, val)
    assert torch.allclose(mat_row, col)
    assert torch.allclose(mat_col, row)


@pytest.mark.parametrize("row", [[0, 0, 1, 2], (0, 1, 2, 4)])
@pytest.mark.parametrize("col", [(0, 1, 2, 2), (1, 3, 3, 4)])
@pytest.mark.parametrize("nz_dim", [None, 2])
@pytest.mark.parametrize("shape", [(5, 5), (6, 7)])
def test_torch_sparse_coo_conversion(row, col, nz_dim, shape):
    dev = F.ctx()
    row = torch.tensor(row).to(dev)
    col = torch.tensor(col).to(dev)
    indices = torch.stack([row, col])
    torch_sparse_shape = shape
    val_shape = (row.shape[0],)
    if nz_dim is not None:
        torch_sparse_shape += (nz_dim,)
        val_shape += (nz_dim,)
    val = torch.randn(val_shape).to(dev)
    torch_sparse_coo = torch.sparse_coo_tensor(indices, val, torch_sparse_shape)
    spmat = from_torch_sparse(torch_sparse_coo)

    def _assert_spmat_equal_to_torch_sparse_coo(spmat, torch_sparse_coo):
        assert torch_sparse_coo.layout == torch.sparse_coo
        # Use .data_ptr() to check whether indices and values are on the same
        # memory address
        assert (
            spmat.indices().data_ptr() == torch_sparse_coo._indices().data_ptr()
        )
        assert spmat.val.data_ptr() == torch_sparse_coo._values().data_ptr()
        assert spmat.shape == torch_sparse_coo.shape[:2]

    _assert_spmat_equal_to_torch_sparse_coo(spmat, torch_sparse_coo)
    torch_sparse_coo = to_torch_sparse_coo(spmat)
    _assert_spmat_equal_to_torch_sparse_coo(spmat, torch_sparse_coo)


@pytest.mark.parametrize("indptr", [(0, 0, 1, 4), (0, 1, 2, 4)])
@pytest.mark.parametrize("indices", [(0, 1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("shape", [(3, 5), (3, 7)])
def test_torch_sparse_csr_conversion(indptr, indices, shape):
    dev = F.ctx()
    indptr = torch.tensor(indptr).to(dev)
    indices = torch.tensor(indices).to(dev)
    torch_sparse_shape = shape
    val_shape = (indices.shape[0],)
    val = torch.randn(val_shape).to(dev)
    torch_sparse_csr = _torch_sparse_csr_tensor(
        indptr, indices, val, torch_sparse_shape
    )
    spmat = from_torch_sparse(torch_sparse_csr)

    def _assert_spmat_equal_to_torch_sparse_csr(spmat, torch_sparse_csr):
        indptr, indices, value_indices = spmat.csr()
        assert torch_sparse_csr.layout == torch.sparse_csr
        assert value_indices is None
        # Use .data_ptr() to check whether indices and values are on the same
        # memory address
        assert indptr.data_ptr() == torch_sparse_csr.crow_indices().data_ptr()
        assert indices.data_ptr() == torch_sparse_csr.col_indices().data_ptr()
        assert spmat.val.data_ptr() == torch_sparse_csr.values().data_ptr()
        assert spmat.shape == torch_sparse_csr.shape[:2]

    _assert_spmat_equal_to_torch_sparse_csr(spmat, torch_sparse_csr)
    torch_sparse_csr = to_torch_sparse_csr(spmat)
    _assert_spmat_equal_to_torch_sparse_csr(spmat, torch_sparse_csr)


@pytest.mark.parametrize("indptr", [(0, 0, 1, 4), (0, 1, 2, 4)])
@pytest.mark.parametrize("indices", [(0, 1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("shape", [(8, 3), (5, 3)])
def test_torch_sparse_csc_conversion(indptr, indices, shape):
    dev = F.ctx()
    indptr = torch.tensor(indptr).to(dev)
    indices = torch.tensor(indices).to(dev)
    torch_sparse_shape = shape
    val_shape = (indices.shape[0],)
    val = torch.randn(val_shape).to(dev)
    torch_sparse_csc = torch.sparse_csc_tensor(
        indptr, indices, val, torch_sparse_shape
    )
    spmat = from_torch_sparse(torch_sparse_csc)

    def _assert_spmat_equal_to_torch_sparse_csc(spmat, torch_sparse_csc):
        indptr, indices, value_indices = spmat.csc()
        assert torch_sparse_csc.layout == torch.sparse_csc
        assert value_indices is None
        # Use .data_ptr() to check whether indices and values are on the same
        # memory address
        assert indptr.data_ptr() == torch_sparse_csc.ccol_indices().data_ptr()
        assert indices.data_ptr() == torch_sparse_csc.row_indices().data_ptr()
        assert spmat.val.data_ptr() == torch_sparse_csc.values().data_ptr()
        assert spmat.shape == torch_sparse_csc.shape[:2]

    _assert_spmat_equal_to_torch_sparse_csc(spmat, torch_sparse_csc)
    torch_sparse_csc = to_torch_sparse_csc(spmat)
    _assert_spmat_equal_to_torch_sparse_csc(spmat, torch_sparse_csc)


### Diag foramt related tests ###


@pytest.mark.parametrize("val_shape", [(3,), (3, 2)])
@pytest.mark.parametrize("mat_shape", [None, (3, 5), (5, 3)])
def test_diag(val_shape, mat_shape):
    ctx = F.ctx()
    # creation
    val = torch.randn(val_shape).to(ctx)
    mat = diag(val, mat_shape)

    # val, shape attributes
    assert torch.allclose(mat.val, val)
    if mat_shape is None:
        mat_shape = (val_shape[0], val_shape[0])
    assert mat.shape == mat_shape

    val = torch.randn(val_shape).to(ctx)

    # nnz
    assert mat.nnz == val.shape[0]
    # dtype
    assert mat.dtype == val.dtype
    # device
    assert mat.device == val.device

    # row, col, val
    edge_index = torch.arange(len(val)).to(mat.device)
    row, col = mat.coo()
    val = mat.val
    assert torch.allclose(row, edge_index)
    assert torch.allclose(col, edge_index)
    assert torch.allclose(val, val)


@pytest.mark.parametrize("shape", [(3, 3), (3, 5), (5, 3)])
@pytest.mark.parametrize("d", [None, 2])
def test_identity(shape, d):
    ctx = F.ctx()
    # creation
    mat = identity(shape, d)
    # shape
    assert mat.shape == shape
    # val
    len_val = min(shape)
    if d is None:
        val_shape = len_val
    else:
        val_shape = (len_val, d)
    val = torch.ones(val_shape)
    assert torch.allclose(val, mat.val)


@pytest.mark.parametrize("val_shape", [(3,), (3, 2)])
@pytest.mark.parametrize("mat_shape", [None, (3, 5), (5, 3)])
def test_diag_matrix_transpose(val_shape, mat_shape):
    ctx = F.ctx()
    val = torch.randn(val_shape).to(ctx)
    mat = diag(val, mat_shape).transpose()

    assert torch.allclose(mat.val, val)
    if mat_shape is None:
        mat_shape = (val_shape[0], val_shape[0])
    assert mat.shape == mat_shape[::-1]
