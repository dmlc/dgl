import pytest
import torch
import sys

from dgl.mock_sparse2 import create_from_coo, create_from_csr, create_from_csc

# FIXME: Skipping tests on win.
if not sys.platform.startswith("linux"):
    pytest.skip("skipping tests on win", allow_module_level=True)

@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("row", [[0, 0, 1, 2], (0, 1, 2, 4)])
@pytest.mark.parametrize("col", [(0, 1, 2, 2), (1, 3, 3, 4)])
@pytest.mark.parametrize("shape", [None, (3, 5), (5, 3)])
def test_create_from_coo(dense_dim, row, col, shape):
    # Skip invalid matrices
    if shape is not None and (
        max(row) >= shape[0] or max(col) >= shape[1]
    ):
        return

    val_shape = (len(row),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    val = torch.randn(val_shape)
    row = torch.tensor(row)
    col = torch.tensor(col)
    mat = create_from_coo(row, col, val, shape)

    if shape is None:
        shape = (torch.max(row).item() + 1, torch.max(col).item() + 1)
    
    mat_row, mat_col, mat_val = mat.coo()

    assert mat.shape == shape
    assert mat.nnz == row.numel()
    assert mat.dtype == val.dtype
    assert torch.allclose(mat_val, val)
    assert torch.allclose(mat_row, row)
    assert torch.allclose(mat_col, col)


@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("indptr", [[0, 0, 1, 4], (0, 1, 2, 4)])
@pytest.mark.parametrize("indices", [(0, 1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("shape", [None, (3, 5)])
def test_create_from_csr(dense_dim, indptr, indices, shape):
    val_shape = (len(indices),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    val = torch.randn(val_shape)
    indptr = torch.tensor(indptr)
    indices = torch.tensor(indices)
    mat = create_from_csr(indptr, indices, val, shape)

    if shape is None:
        shape = (indptr.numel() - 1, torch.max(indices).item() + 1)

    assert mat.device == val.device
    assert mat.shape == shape
    assert mat.nnz == indices.numel()
    assert mat.dtype == val.dtype
    mat_indptr, mat_indices, mat_val = mat.csr()
    assert torch.allclose(mat_indptr, indptr)
    assert torch.allclose(mat_indices, indices)
    assert torch.allclose(mat_val, val)

@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("indptr", [[0, 0, 1, 4], (0, 1, 2, 4)])
@pytest.mark.parametrize("indices", [(0, 1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("shape", [None, (5, 3)])
def test_create_from_csc(dense_dim, indptr, indices, shape):
    val_shape = (len(indices),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    val = torch.randn(val_shape)
    indptr = torch.tensor(indptr)
    indices = torch.tensor(indices)
    mat = create_from_csc(indptr, indices, val, shape)

    if shape is None:
        shape = (torch.max(indices).item() + 1, indptr.numel() - 1)

    assert mat.device == val.device
    assert mat.shape == shape
    assert mat.nnz == indices.numel()
    assert mat.dtype == val.dtype
    mat_indptr, mat_indices, mat_val = mat.csc()
    assert torch.allclose(mat_indptr, indptr)
    assert torch.allclose(mat_indices, indices)
    assert torch.allclose(mat_val, val)
