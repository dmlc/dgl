""" dgl c++ class binding"""
import os
import torch

package_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
so_path = os.path.join(package_path, 'tensoradaptor/pytorch/libdgl_sparse.so')
torch.classes.load_library(so_path)

_SparseMatrix = torch.classes.dgl_sparse.SparseMatrix
