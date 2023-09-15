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


@pytest.mark.skipif(F.ctx().type == "cuda", reason="GPU not support now")
@pytest.mark.parametrize(
    "create_func", [rand_diag, rand_csr, rand_csc, rand_coo]
)
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("index", [None, (1, 2), (0, 0, 2), (2, 0)])
def test_compact(create_func, dim, index):
    ctx = F.ctx()
    shape = (3, 3)
    ans_idx = []
    if index is not None:
        ans_idx = list(index)
        index = torch.tensor(index).to(ctx)

    A = create_func(shape, 6, ctx)

    print(A)
    print()
    print(index)

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

    print(A_dense, index)
    print(A_dense_select)
    print(A_compact_dense)

    assert A_compact_dense.shape == A_dense_select.shape
    assert torch.allclose(A_compact_dense, A_dense_select)
    assert torch.allclose(ans_idx, ret_id)
