import pytest
import torch

import backend as F

from dgl.mock_sparse import diag, identity, DiagMatrix

@pytest.mark.parametrize('val_shape', [(3,), (3, 2)])
@pytest.mark.parametrize('mat_shape', [None, (3, 5), (5, 3)])
def test_diag(val_shape, mat_shape):
    # creation
    val = torch.randn(val_shape).to(F.ctx())
    mat = diag(val, mat_shape)

    # val, shape attributes
    assert torch.allclose(mat.val, val)
    if mat_shape is None:
        mat_shape = (val_shape[0], val_shape[0])
    assert mat.shape == mat_shape

    # __call__
    val = torch.randn(val_shape).to(F.ctx())
    mat = mat(val)
    assert torch.allclose(mat.val, val)

    # nnz
    assert mat.nnz == val.shape[0]
    # dtype
    assert mat.dtype == val.dtype
    # device
    assert mat.device == val.device

    # as_sparse
    sp_mat = mat.as_sparse()
    # shape
    assert sp_mat.shape == mat_shape
    # nnz
    assert sp_mat.nnz == mat.nnz
    # dtype
    assert sp_mat.dtype == mat.dtype
    # device
    assert sp_mat.device == mat.device
    # row, col, val
    edge_index = torch.arange(len(val)).to(mat.device)
    assert torch.allclose(sp_mat.row, edge_index)
    assert torch.allclose(sp_mat.col, edge_index)
    assert torch.allclose(sp_mat.val, val)

@pytest.mark.parametrize('shape', [(3, 3), (3, 5), (5, 3)])
@pytest.mark.parametrize('d', [None, 2])
def test_identity(shape, d):
    # creation
    mat = identity(shape, d, device=F.ctx())
    # type
    assert isinstance(mat, DiagMatrix)
    # shape
    assert mat.shape == shape
    # val
    len_val = min(shape)
    if d is None:
        val_shape = (len_val)
    else:
        val_shape = (len_val, d)
    val = torch.ones(val_shape, device=F.ctx())
    assert torch.allclose(val, mat.val)
