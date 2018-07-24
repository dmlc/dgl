import torch as F
from dgl.array import DGLDenseArray

x = F.randn(100, 10)
a = DGLDenseArray(F)
