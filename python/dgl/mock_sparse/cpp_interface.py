""" DGL sparse library C++ binding"""
import os
import torch

package_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
so_path = os.path.join(package_path, '../tensoradapter/pytorch/libdgl_sparse.so')
torch.classes.load_library(so_path)

SparseMatrix = torch.classes.dgl_sparse.SparseMatrix
create_from_coo = torch.ops.dgl_sparse.create_from_coo
