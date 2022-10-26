import operator

import numpy as np
import pytest
import torch

import dgl
from dgl.mock_sparse.cpp_interface import create_from_coo

parametrize_idtype = pytest.mark.parametrize(
    "idtype", [torch.int32, torch.int64]
)
parametrize_dtype = pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64]
)

@parametrize_idtype
@parametrize_dtype
def test_sparse_op_sparse(idtype, dtype):
    rowA = torch.tensor([1, 0, 2, 7, 1])
    colA = torch.tensor([0, 49, 2, 1, 7])
    valA = torch.rand(len(rowA))
    A = create_from_coo(rowA, colA, valA, (10, 50))
    valB = torch.rand(len(rowA))
    B = create_from_coo(rowA, colA, valB, (10, 50))

    C = A.__add__(B)
    torch.allclose(C.val(), valA + valB)
