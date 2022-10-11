import pytest
import torch
import numpy

from dgl.mock_sparse import create_from_coo


@pytest.mark.skip(reason="no way of currently testing this")
@pytest.mark.parametrize("dense_dim", [None, 2])
@pytest.mark.parametrize("row", [[0, 0, 1, 2], (0, 1, 2, 4)])
@pytest.mark.parametrize("col", [(0, 1, 2, 2), (1, 3, 3, 4)])
@pytest.mark.parametrize("extra_shape", [(0, 1), (2, 1)])
@pytest.mark.parametrize("reduce_type", ['sum', 'smax', 'smin', 'smean'])
@pytest.mark.parametrize("dim", [None, 0, 1])
def test_reduction(dense_dim, row, col, extra_shape, reduce_type, dim):
    mat_shape = (max(row) + 1 + extra_shape[0], max(col) + 1 + extra_shape[1])

    val_shape = (len(row),)
    if dense_dim is not None:
        val_shape += (dense_dim,)

    val = torch.randn(val_shape)
    row = torch.tensor(row)
    col = torch.tensor(col)
    mat = create_from_coo(row, col, val, mat_shape)

    reduce_func = getattr(mat, reduce_type)
    reduced = reduce_func(dim)

    def calc_expected(row, col, val, mat_shape, reduce_type, dim):
        def reduce_func(reduce_type, lhs, rhs):
            if lhs is None:
                return rhs
            if reduce_type == 'sum' or reduce_type == 'smean':
                return lhs + rhs
            if reduce_type == 'smax':
                return numpy.maximum(lhs, rhs)
            if reduce_type == 'smin':
                return numpy.minimum(lhs, rhs)

        val = val.numpy()
        row = row.numpy()
        col = col.numpy()
        if dim is None:
            reduced = None
            for i in range(val.shape[0]):
                reduced = reduce_func(reduce_type, reduced, val[i])
            if reduced is None:
                reduced = numpy.zeros(val.shape[1:])
            if reduce_type == 'smean':
                reduced = reduced / val.shape[0]
            return reduced

        reduced_shape = (mat_shape[0] if dim == 1 else mat_shape[1])
        reduced = [None] * reduced_shape
        count = [0] * reduced_shape
        for i, (r, c) in enumerate(zip(row, col)):
            axis = r if dim == 1 else c
            reduced[axis] = reduce_func(reduce_type, reduced[axis], val[i])
            count[axis] += 1

        for i in range(reduced_shape):
            if count[i] == 0:
                reduced[i] = numpy.zeros(val.shape[1:])
            else:
                if reduce_type == 'smean':
                    reduced[i] /= count[i]
        return numpy.stack(reduced, axis=0)
    expected = calc_expected(row, col, val, mat_shape, reduce_type, dim)

    assert torch.allclose(reduced, torch.tensor(expected).float())
