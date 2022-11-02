import pytest
import torch
import sys

if sys.platform.startswith("linux"):
    from dgl.mock_sparse2 import create_from_coo


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Sparse library only supports linux for now",
)
@pytest.mark.parametrize("dense_dim", [None, 4])
@pytest.mark.parametrize("row", [[0, 0, 1, 2], (0, 1, 2, 4)])
@pytest.mark.parametrize("col", [(0, 1, 2, 2), (1, 3, 3, 4)])
@pytest.mark.parametrize("mat_shape", [(3, 5), (5, 3)])
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

    mat_row, mat_col, mat_val = mat.coo()
    assert tuple(mat.shape) == mat_shape
    assert torch.allclose(mat_val, val)
    assert torch.allclose(mat_row, row)
    assert torch.allclose(mat_col, col)
