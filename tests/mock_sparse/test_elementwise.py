import numpy as np
import pytest
import dgl
import dgl.backend as F
import torch
from dgl.mock_dgl_sparse.python.sp_matrix import SparseMatrix

def create_sp_matrix(M, N, nnz):
	row = torch.tensor([1, 1, 2])
	col = torch.tensor([2, 4, 3])
	val = torch.tensor([3 ,4, 5])
	return SparseMatrix(row, col, val)

# TODO(Israt): Add parameterized idtype and dtype
def test_sparse_op_sparse():
   	A = create_sp_matrix(500, 500, 500)
   	print("shape", A.shape)
   	print("nnz", A.nnz)
   	print('dtype:', A.dtype)
   	print('device:', A.device)
   	print('rows:', A.row)
   	print('cols:', A.col)
   	print('vals:', A.val)


if __name__ == '__main__':
    test_sparse_op_sparse()

 
