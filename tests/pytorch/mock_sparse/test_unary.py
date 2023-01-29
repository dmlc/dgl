import pytest
import torch

import backend as F

from dgl.convert import graph
from dgl.mock_sparse import diag, create_from_coo
from dgl.ops import edge_softmax

@pytest.mark.parametrize('val_shape', [(3,), (3, 2)])
@pytest.mark.parametrize('mat_shape', [(3, 3), (5, 3)])
def test_neg_diag(val_shape, mat_shape):
    val = torch.randn(val_shape).to(F.ctx())
    mat = diag(val, mat_shape)
    neg_mat = -mat
    assert neg_mat.shape == mat.shape
    assert torch.allclose(-mat.val, neg_mat.val)

def test_inv_diag():
    val = torch.randn(3).to(F.ctx())
    mat = diag(val, (3, 3))
    inv_mat = mat.inv()
    assert inv_mat.shape == mat.shape
    assert torch.allclose(1. / mat.val, inv_mat.val)

@pytest.mark.parametrize('val_shape', [(3,), (3, 2)])
@pytest.mark.parametrize('mat_shape', [(3, 3), (5, 3)])
def test_softmax_diag(val_shape, mat_shape):
    val = torch.randn(val_shape).to(F.ctx())
    mat = diag(val, mat_shape)
    softmax_mat = mat.softmax()
    assert softmax_mat.shape == mat.shape
    assert torch.allclose(softmax_mat.val, torch.ones_like(mat.val))

@pytest.mark.parametrize('val_shape', [(3,), (3, 2)])
@pytest.mark.parametrize('mat_shape', [(4, 4), (5, 4)])
def test_neg_sp(val_shape, mat_shape):
    device = F.ctx()
    row = torch.tensor([1, 1, 3]).to(device)
    col = torch.tensor([1, 2, 3]).to(device)
    val = torch.randn(val_shape).to(device)
    mat = create_from_coo(row, col, val, mat_shape)
    neg_mat = -mat
    assert neg_mat.shape == mat.shape
    assert torch.allclose(-mat.val, neg_mat.val)

def test_inv_sp():
    device = F.ctx()
    row = torch.tensor([0, 1, 1]).to(device)
    col = torch.tensor([0, 0, 1]).to(device)
    val = torch.tensor([1., 1., 2.]).to(device)
    mat = create_from_coo(row, col, val)
    inv_mat = mat.inv()
    assert inv_mat.shape == mat.shape
    assert torch.allclose(torch.tensor([1., -0.5, 0.5]).to(device), inv_mat.val)

@pytest.mark.parametrize('val_shape', [(4,), (4, 2)])
def test_softmax_sp(val_shape):
    device = F.ctx()
    row = torch.tensor([0, 0, 1, 2]).to(device)
    col = torch.tensor([1, 2, 2, 0]).to(device)
    val = torch.randn(val_shape).to(device)
    mat = create_from_coo(row, col, val)
    result = mat.softmax()
    assert result.shape == mat.shape
    g = graph((mat.col, mat.row))
    assert torch.allclose(result.val, edge_softmax(g, mat.val))
