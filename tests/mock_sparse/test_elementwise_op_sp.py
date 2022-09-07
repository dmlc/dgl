import numpy as np
import pytest
import dgl
import dgl.backend as F
import torch
import numpy
import operator
from dgl.mock_dgl_sparse.python import elementwise_sparse
from dgl.mock_dgl_sparse.python.sp_matrix import SparseMatrix
parametrize_idtype = pytest.mark.parametrize("idtype", [F.int32, F.int64])
parametrize_dtype = pytest.mark.parametrize('dtype', [F.float32, F.float64])


ops = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}

@parametrize_idtype
@parametrize_dtype
def test_sparse_op_sparse(idtype, dtype):
    row = torch.randint(1, 500, (100,))
    col = torch.randint(1, 500, (100,))
    val = torch.rand(100)
    A = SparseMatrix(row, col, val)

    def _test(A, op):
        out_torch = eval("A.val " + op + " A.val")
        out_spMat = eval("A " + op + " A").val
        assert np.allclose(out_torch, out_spMat, rtol=1e-4, atol=1e-4)
        w1 = torch.rand(100)
        w2 = torch.rand(100)
        out_torch2 = eval("w1 " + op + " w2")
        out_spMat2 = eval("A(w1) " + op + " A(w2)").val
        assert np.allclose(out_torch2, out_spMat2, rtol=1e-4, atol=1e-4)

    for op in ops:
        _test(A, op)

@parametrize_idtype
@parametrize_dtype
@pytest.mark.parametrize('v_scalar', [2, 2.5])
def test_sparse_op_scalar(idtype, dtype, v_scalar):
    row = torch.randint(1, 500, (100,))
    col = torch.randint(1, 500, (100,))
    val = torch.rand(100)
    A = SparseMatrix(row, col, val)
    out_torch = A.val * v_scalar
    out_spMat = A * v_scalar
    assert np.allclose(out_torch, out_spMat.val, rtol=1e-4, atol=1e-4)
    out_torch = A.val / v_scalar
    out_spMat = A / v_scalar
    assert np.allclose(out_torch, out_spMat.val, rtol=1e-4, atol=1e-4)
    out_torch = pow(A.val, v_scalar)
    out_spMat = pow(A, v_scalar)
    assert np.allclose(out_torch, out_spMat.val, rtol=1e-4, atol=1e-4)

@parametrize_idtype
@parametrize_dtype
@pytest.mark.parametrize('v_scalar', [2, 2.5])
def test_scalar_op_sparse(idtype, dtype, v_scalar):
    row = torch.randint(1, 500, (100,))
    col = torch.randint(1, 500, (100,))
    val = torch.rand(100)
    A = SparseMatrix(row, col, val)
    out_torch = v_scalar * A.val
    out_spMat = v_scalar * A
    assert np.allclose(out_torch, out_spMat.val, rtol=1e-4, atol=1e-4)

if __name__ == '__main__':
    test_sparse_op_sparse()
    test_sparse_op_scalar()
    test_scalar_op_sparse()

