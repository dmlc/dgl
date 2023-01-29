import sys
import unittest

import backend as F
import pytest
import torch

from dgl.sparse import diag, DiagMatrix, identity

# TODO(#4818): Skipping tests on win.
if not sys.platform.startswith("linux"):
    pytest.skip("skipping tests on win", allow_module_level=True)


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

    # as_sparse
    sp_mat = mat.to_sparse()
    # shape
    assert tuple(sp_mat.shape) == mat_shape
    # nnz
    assert sp_mat.nnz == mat.nnz
    # dtype
    assert sp_mat.dtype == mat.dtype
    # device
    assert sp_mat.device == mat.device
    # row, col, val
    edge_index = torch.arange(len(val)).to(mat.device)
    row, col = sp_mat.coo()
    val = sp_mat.val
    assert torch.allclose(row, edge_index)
    assert torch.allclose(col, edge_index)
    assert torch.allclose(val, val)


@pytest.mark.parametrize("shape", [(3, 3), (3, 5), (5, 3)])
@pytest.mark.parametrize("d", [None, 2])
def test_identity(shape, d):
    ctx = F.ctx()
    # creation
    mat = identity(shape, d)
    # type
    assert isinstance(mat, DiagMatrix)
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


def test_print():
    ctx = F.ctx()

    # basic
    val = torch.tensor([1.0, 1.0, 2.0]).to(ctx)
    A = diag(val)
    print(A)

    # vector-shape non zero
    val = torch.randn(3, 2).to(ctx)
    A = diag(val)
    print(A)


@unittest.skipIf(
    F._default_context_str == "cpu",
    reason="Device conversions don't need to be tested on CPU.",
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_to_device(device):
    val = torch.randn(3)
    mat_shape = (3, 4)
    mat = diag(val, mat_shape)

    target_val = mat.val.to(device)
    mat2 = mat.to(device=device)
    assert mat2.shape == mat.shape
    assert torch.allclose(mat2.val, target_val)

    mat2 = getattr(mat, device)()
    assert mat2.shape == mat.shape
    assert torch.allclose(mat2.val, target_val)


@pytest.mark.parametrize(
    "dtype", [torch.float, torch.double, torch.int, torch.long]
)
def test_to_dtype(dtype):
    val = torch.randn(3)
    mat_shape = (3, 4)
    mat = diag(val, mat_shape)

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
