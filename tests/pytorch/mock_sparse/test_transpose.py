import pytest
import torch

from dgl.mock_sparse import diag, create_from_coo


@pytest.mark.parametrize("val_shape", [(3,), (3, 2)])
@pytest.mark.parametrize("mat_shape", [None, (3, 5), (5, 3)])
def test_diag_matrix_transpose(val_shape, mat_shape):
    val = torch.randn(val_shape)
    mat = diag(val, mat_shape).transpose()

    assert torch.allclose(mat.val, val)
    if mat_shape is None:
        mat_shape = (val_shape[0], val_shape[0])
    assert mat.shape == mat_shape[::-1]


@pytest.mark.parametrize("dense_dim", [None, 2])
@pytest.mark.parametrize("row", [[0, 0, 1, 2], (0, 1, 2, 4)])
@pytest.mark.parametrize("col", [(0, 1, 2, 2), (1, 3, 3, 4)])
@pytest.mark.parametrize("extra_shape", [(0, 1), (2, 1)])
def test_sparse_matrix_transpose(dense_dim, row, col, extra_shape):
    mat_shape = (max(row) + 1 + extra_shape[0], max(col) + 1 + extra_shape[1])
    val_shape = (len(row),)
    if dense_dim is not None:
        val_shape += (dense_dim,)
    val = torch.randn(val_shape)
    row = torch.tensor(row)
    col = torch.tensor(col)
    mat = create_from_coo(row, col, val, mat_shape).transpose()

    assert mat.shape == mat_shape[::-1]
    assert torch.allclose(mat.val, val)
    assert torch.allclose(mat.row, col)
    assert torch.allclose(mat.col, row)