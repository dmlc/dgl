import operator
import sys

import backend as F

import dgl.sparse
import pytest
import torch
from dgl.sparse import add, diag, from_coo, from_csc, from_csr, sub

# TODO(#4818): Skipping tests on win.
if not sys.platform.startswith("linux"):
    pytest.skip("skipping tests on win", allow_module_level=True)


@pytest.mark.parametrize("val_shape", [(), (2,)])
@pytest.mark.parametrize("opname", ["add", "sub"])
def test_addsub_coo(val_shape, opname):
    op = getattr(operator, opname)
    func = getattr(dgl.sparse, opname)
    ctx = F.ctx()
    row = torch.tensor([1, 0, 2]).to(ctx)
    col = torch.tensor([0, 3, 2]).to(ctx)
    val = torch.randn(row.shape + val_shape).to(ctx)
    A = from_coo(row, col, val)

    row = torch.tensor([1, 0]).to(ctx)
    col = torch.tensor([0, 2]).to(ctx)
    val = torch.randn(row.shape + val_shape).to(ctx)
    B = from_coo(row, col, val, shape=A.shape)

    C1 = op(A, B).dense()
    C2 = func(A, B).dense()
    dense_C = op(A.dense(), B.dense())

    assert torch.allclose(dense_C, C1)
    assert torch.allclose(dense_C, C2)

    with pytest.raises(TypeError):
        op(A, 2)
    with pytest.raises(TypeError):
        op(2, A)


@pytest.mark.parametrize("val_shape", [(), (2,)])
@pytest.mark.parametrize("opname", ["add", "sub"])
def test_addsub_csr(val_shape, opname):
    op = getattr(operator, opname)
    func = getattr(dgl.sparse, opname)
    ctx = F.ctx()
    indptr = torch.tensor([0, 1, 2, 3]).to(ctx)
    indices = torch.tensor([3, 0, 2]).to(ctx)
    val = torch.randn(indices.shape + val_shape).to(ctx)
    A = from_csr(indptr, indices, val)

    indptr = torch.tensor([0, 1, 2, 2]).to(ctx)
    indices = torch.tensor([2, 0]).to(ctx)
    val = torch.randn(indices.shape + val_shape).to(ctx)
    B = from_csr(indptr, indices, val, shape=A.shape)

    C1 = op(A, B).dense()
    C2 = func(A, B).dense()
    dense_C = op(A.dense(), B.dense())

    assert torch.allclose(dense_C, C1)
    assert torch.allclose(dense_C, C2)

    with pytest.raises(TypeError):
        op(A, 2)
    with pytest.raises(TypeError):
        op(2, A)


@pytest.mark.parametrize("val_shape", [(), (2,)])
@pytest.mark.parametrize("opname", ["add", "sub"])
def test_addsub_csc(val_shape, opname):
    op = getattr(operator, opname)
    func = getattr(dgl.sparse, opname)
    ctx = F.ctx()
    indptr = torch.tensor([0, 1, 1, 2, 3]).to(ctx)
    indices = torch.tensor([1, 2, 0]).to(ctx)
    val = torch.randn(indices.shape + val_shape).to(ctx)
    A = from_csc(indptr, indices, val)

    indptr = torch.tensor([0, 1, 1, 2, 2]).to(ctx)
    indices = torch.tensor([1, 0]).to(ctx)
    val = torch.randn(indices.shape + val_shape).to(ctx)
    B = from_csc(indptr, indices, val, shape=A.shape)

    C1 = op(A, B).dense()
    C2 = func(A, B).dense()
    dense_C = op(A.dense(), B.dense())

    assert torch.allclose(dense_C, C1)
    assert torch.allclose(dense_C, C2)

    with pytest.raises(TypeError):
        op(A, 2)
    with pytest.raises(TypeError):
        op(2, A)


@pytest.mark.parametrize("val_shape", [(), (2,)])
@pytest.mark.parametrize("opname", ["add", "sub"])
def test_addsub_diag(val_shape, opname):
    op = getattr(operator, opname)
    func = getattr(dgl.sparse, opname)
    ctx = F.ctx()
    shape = (3, 4)
    val_shape = (shape[0],) + val_shape
    D1 = diag(torch.randn(val_shape).to(ctx), shape=shape)
    D2 = diag(torch.randn(val_shape).to(ctx), shape=shape)

    C1 = op(D1, D2).dense()
    C2 = func(D1, D2).dense()
    dense_C = op(D1.dense(), D2.dense())

    assert torch.allclose(dense_C, C1)
    assert torch.allclose(dense_C, C2)

    with pytest.raises(TypeError):
        op(D1, 2)
    with pytest.raises(TypeError):
        op(2, D1)


@pytest.mark.parametrize("val_shape", [(), (2,)])
def test_add_sparse_diag(val_shape):
    ctx = F.ctx()
    row = torch.tensor([1, 0, 2]).to(ctx)
    col = torch.tensor([0, 3, 2]).to(ctx)
    val = torch.randn(row.shape + val_shape).to(ctx)
    A = from_coo(row, col, val)

    shape = (3, 4)
    val_shape = (shape[0],) + val_shape
    D = diag(torch.randn(val_shape).to(ctx), shape=shape)

    sum1 = (A + D).dense()
    sum2 = (D + A).dense()
    sum3 = add(A, D).dense()
    sum4 = add(D, A).dense()
    dense_sum = A.dense() + D.dense()

    assert torch.allclose(dense_sum, sum1)
    assert torch.allclose(dense_sum, sum2)
    assert torch.allclose(dense_sum, sum3)
    assert torch.allclose(dense_sum, sum4)


@pytest.mark.parametrize("val_shape", [(), (2,)])
def test_sub_sparse_diag(val_shape):
    ctx = F.ctx()
    row = torch.tensor([1, 0, 2]).to(ctx)
    col = torch.tensor([0, 3, 2]).to(ctx)
    val = torch.randn(row.shape + val_shape).to(ctx)
    A = from_coo(row, col, val)

    shape = (3, 4)
    val_shape = (shape[0],) + val_shape
    D = diag(torch.randn(val_shape).to(ctx), shape=shape)

    diff1 = (A - D).dense()
    diff2 = (D - A).dense()
    diff3 = sub(A, D).dense()
    diff4 = sub(D, A).dense()
    dense_diff = A.dense() - D.dense()

    assert torch.allclose(dense_diff, diff1)
    assert torch.allclose(dense_diff, -diff2)
    assert torch.allclose(dense_diff, diff3)
    assert torch.allclose(dense_diff, -diff4)


@pytest.mark.parametrize("op", ["mul", "truediv", "pow"])
def test_error_op_sparse_diag(op):
    ctx = F.ctx()
    row = torch.tensor([1, 0, 2]).to(ctx)
    col = torch.tensor([0, 3, 2]).to(ctx)
    val = torch.randn(row.shape).to(ctx)
    A = from_coo(row, col, val)

    shape = (3, 4)
    D = diag(torch.randn(row.shape[0]).to(ctx), shape=shape)

    with pytest.raises(TypeError):
        getattr(operator, op)(A, D)
    with pytest.raises(TypeError):
        getattr(operator, op)(D, A)
