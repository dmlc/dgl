import operator
import sys

import backend as F

import dgl.mock_sparse2 as dglsp
import pytest
import torch
import doctest

dgl_op_map = {
    "sum": "sum",
    "amin": "smin",
    "amax": "smax",
    "mean": "smean",
    "prod": "sprod",
}
default_entry = {
    "sum": 0,
    "amin": float("inf"),
    "amax": float("-inf"),
    "mean": 0,
    "prod": 1,
}
binary_op_map = {
    "sum": operator.add,
    "amin": torch.min,
    "amax": torch.max,
    "mean": operator.add,
    "prod": operator.mul,
}

NUM_ROWS = 10
NUM_COLS = 15


def _coalesce_dense(row, col, val, nrows, ncols, op):
    M = torch.zeros(NUM_ROWS, NUM_COLS, device=F.ctx())
    A2 = torch.full(
        (NUM_ROWS, NUM_COLS, 20) + val.shape[1:],
        default_entry[op],
        device=F.ctx(),
        dtype=val.dtype,
    )
    A2 = torch.index_put(A2, (row, col, torch.arange(20)), val)
    for i in range(20):
        M[row[i], col[i]] += 1
    if op == "mean":
        A2 = A2.sum(2)
    else:
        A2 = getattr(A2, op)(2)
    M = M.view(NUM_ROWS, NUM_COLS, *([1] * (val.dim() - 1)))
    return A2, M


# Add docstring tests of dglsp.reduction to unit tests
@pytest.mark.parametrize(
    "func", ["reduce", "sum", "smin", "smax", "sprod", "smean"]
)
def test_docstring(func):
    globs = {"torch": torch, "dglsp": dglsp}
    runner = doctest.DebugRunner()
    finder = doctest.DocTestFinder()
    obj = getattr(dglsp, func)
    for test in finder.find(obj, func, globs=globs):
        runner.run(test)


@pytest.mark.parametrize("shape", [(20,), (20, 20), (20, 20, 20)])
@pytest.mark.parametrize("op", ["sum", "amin", "amax", "mean", "prod"])
@pytest.mark.parametrize("use_reduce", [False, True])
def test_reduce_all(shape, op, use_reduce):
    row = torch.randint(0, NUM_ROWS, (20,), device=F.ctx())
    col = torch.randint(0, NUM_COLS, (20,), device=F.ctx())
    val = torch.randn(*shape, device=F.ctx())
    val2 = val.clone()
    val = val.requires_grad_()
    val2 = val2.requires_grad_()
    A = dglsp.create_from_coo(row, col, val, shape=(NUM_ROWS, NUM_COLS))

    A2, M = _coalesce_dense(row, col, val2, NUM_ROWS, NUM_COLS, op)

    if not use_reduce:
        output = getattr(A, dgl_op_map[op])()
    else:
        output = A.reduce(rtype=dgl_op_map[op])

    if op == "mean":
        output2 = A2.sum((0, 1)) / M.sum()
    elif op == "prod":
        output2 = A2.prod(0).prod(0)  # prod() does not support tuple of dims
    else:
        output2 = getattr(A2, op)((0, 1))
    assert (output - output2).abs().max() < 1e-4

    head = torch.randn(*output.shape).to(val) if output.dim() > 0 else None
    output.backward(head)
    output2.backward(head)
    assert (val.grad - val2.grad).abs().max() < 1e-4


@pytest.mark.parametrize("shape", [(20,), (20, 20), (20, 20, 20)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("empty_nnz", [False, True])
@pytest.mark.parametrize("op", ["sum", "amin", "amax", "mean", "prod"])
@pytest.mark.parametrize("use_reduce", [False, True])
def test_reduce_along(shape, dim, empty_nnz, op, use_reduce):
    row = torch.randint(0, NUM_ROWS, (20,), device=F.ctx())
    col = torch.randint(0, NUM_COLS, (20,), device=F.ctx())
    if dim == 0:
        mask = torch.bincount(col, minlength=NUM_COLS) == 0
    else:
        mask = torch.bincount(row, minlength=NUM_ROWS) == 0
    val = torch.randn(*shape, device=F.ctx())
    val2 = val.clone()
    val = val.requires_grad_()
    val2 = val2.requires_grad_()
    
    # empty_nnz controls whether at least one column or one row has no
    # non-zero entry.
    if empty_nnz:
        row[row == 0] = 1
        col[col == 0] = 1

    A = dglsp.create_from_coo(row, col, val, shape=(NUM_ROWS, NUM_COLS))

    A2, M = _coalesce_dense(row, col, val2, NUM_ROWS, NUM_COLS, op)

    if not use_reduce:
        output = getattr(A, dgl_op_map[op])(dim)
    else:
        output = A.reduce(dim=dim, rtype=dgl_op_map[op])

    if op == "mean":
        output2 = A2.sum(dim) / M.sum(dim)
    else:
        output2 = getattr(A2, op)(dim)
    zero_entry_idx = (M.sum(dim) != 0).nonzero(as_tuple=True)[0]
    output3 = torch.index_put(
        torch.zeros_like(output2), (zero_entry_idx,), output2[zero_entry_idx]
    )
    assert (output - output3).abs().max() < 1e-4

    head = torch.randn(*output.shape).to(val) if output.dim() > 0 else None
    output.backward(head)
    output3.backward(head)
    assert (val.grad - val2.grad).abs().max() < 1e-4
