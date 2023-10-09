import backend as F
import pytest
import torch

from .utils import (
    rand_coo,
    rand_csc,
    rand_csr,
    rand_diag,
    sparse_matrix_to_dense,
)


@pytest.mark.parametrize(
    "create_func", [rand_diag, rand_csr, rand_csc, rand_coo]
)
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("index", [None, (1, 3), (4, 0, 2)])
def test_compact(create_func, dim, index):
    ctx = F.ctx()
    shape = (5, 5)
    ans_idx = []
    if index is not None:
        ans_idx = list(dict.fromkeys(index))
        index = torch.tensor(index).to(ctx)

    A = create_func(shape, 8, ctx)

    A_compact, ret_id = A.compact(dim, index)
    A_compact_dense = sparse_matrix_to_dense(A_compact)

    A_dense = sparse_matrix_to_dense(A)

    for i in range(shape[dim]):
        if dim == 0:
            row = list(A_dense[i, :].nonzero().reshape(-1))
        else:
            row = list(A_dense[:, i].nonzero().reshape(-1))
        if (i not in list(ans_idx)) and len(row) > 0:
            ans_idx.append(i)
    if len(ans_idx):
        ans_idx = torch.tensor(ans_idx).to(ctx)
    A_dense_select = sparse_matrix_to_dense(A.index_select(dim, ans_idx))

    assert A_compact_dense.shape == A_dense_select.shape
    assert torch.allclose(A_compact_dense, A_dense_select)
    assert torch.allclose(ans_idx, ret_id)
