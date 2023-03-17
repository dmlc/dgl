import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import symbolic_trace


class GCN(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=16):
        super().__init__()

        # Two-layer GCN.
        self.W1 = nn.Linear(in_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, out_size)

    def forward(self, A: dglsp.SparseMatrix, X: torch.Tensor):
        X = dglsp.spmm(A, self.W1(X))
        X = F.relu(X)
        X = dglsp.spmm(A, self.W2(X))
        return X


model = GCN(10, 20)
scripted_model = torch.jit.script(model)
print(scripted_model.code)
