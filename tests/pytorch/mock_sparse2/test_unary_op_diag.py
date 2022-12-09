import pytest
import torch
import sys

import backend as F

from dgl.mock_sparse2 import diag


def test_neg():
    ctx = F.ctx()
    val = torch.arange(3).float().to(ctx)
    D = diag(val)
    neg_D = -D
    assert D.shape == neg_D.shape
    assert torch.allclose(-D.val, neg_D.val)
    assert D.val.device == neg_D.val.device


def test_inv():
    ctx = F.ctx()
    val = torch.arange(1, 4).float().to(ctx)
    D = diag(val)
    inv_D = D.inv()
    assert D.shape == inv_D.shape
    assert torch.allclose(1.0 / D.val, inv_D.val)
    assert D.val.device == inv_D.val.device
