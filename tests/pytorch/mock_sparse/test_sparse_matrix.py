import pytest
import torch

from dgl.mock_sparse import create_from_coo, create_from_csr, create_from_csc


@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("row", [[0, 0, 1, 2], (0, 1, 2, 4)])
@pytest.mark.parametrize("col", [(0, 1, 2, 2), (1, 3, 3, 4)])
@pytest.mark.parametrize("mat_shape", [None, (3, 5), (5, 3)])
def test_create_from_coo(dense_dim, row, col, mat_shape):
    # Skip invalid matrices
    if mat_shape is not None and (
        max(row) >= mat_shape[0] or max(col) >= mat_shape[1]
    ):
        return

    val_shape = (len(row),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    val = torch.randn(val_shape)
    row = torch.tensor(row)
    col = torch.tensor(col)
    mat = create_from_coo(row, col, val, mat_shape)

    if mat_shape is None:
        mat_shape = (torch.max(row).item() + 1, torch.max(col).item() + 1)

    assert mat.shape == mat_shape
    assert mat.nnz == row.numel()
    assert mat.dtype == val.dtype
    assert torch.allclose(mat.val, val)
    assert torch.allclose(mat.row, row)
    assert torch.allclose(mat.col, col)


@pytest.mark.skip(reason="no way of currently testing this")
@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("indptr", [[0, 0, 1, 4], (0, 1, 2, 4)])
@pytest.mark.parametrize("indices", [(0, 1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("mat_shape", [None, (3, 5)])
def test_create_from_csr(dense_dim, indptr, indices, mat_shape):
    val_shape = (len(indices),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    val = torch.randn(val_shape)
    indptr = torch.tensor(indptr)
    indices = torch.tensor(indices)
    mat = create_from_csr(indptr, indices, val, mat_shape)

    if mat_shape is None:
        mat_shape = (indptr.numel() - 1, torch.max(indices).item() + 1)

    assert mat.device == val.device
    assert mat.shape == mat_shape
    assert mat.nnz == indices.numel()
    assert mat.dtype == val.dtype
    assert torch.allclose(mat.val, val)
    deg = torch.diff(indptr)
    row = torch.repeat_interleave(torch.arange(deg.numel()), deg)
    assert torch.allclose(mat.row, row)
    col = indices
    assert torch.allclose(mat.col, col)

@pytest.mark.skip(reason="no way of currently testing this")
@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("indptr", [[0, 0, 1, 4], (0, 1, 2, 4)])
@pytest.mark.parametrize("indices", [(0, 1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("mat_shape", [None, (5, 3)])
def test_create_from_csc(dense_dim, indptr, indices, mat_shape):
    val_shape = (len(indices),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    val = torch.randn(val_shape)
    indptr = torch.tensor(indptr)
    indices = torch.tensor(indices)
    mat = create_from_csc(indptr, indices, val, mat_shape)

    if mat_shape is None:
        mat_shape = (torch.max(indices).item() + 1, indptr.numel() - 1)

    assert mat.device == val.device
    assert mat.shape == mat_shape
    assert mat.nnz == indices.numel()
    assert mat.dtype == val.dtype
    assert torch.allclose(mat.val, val)
    row = indices
    assert torch.allclose(mat.row, row)
    deg = torch.diff(indptr)
    col = torch.repeat_interleave(torch.arange(deg.numel()), deg)
    assert torch.allclose(mat.col, col)

