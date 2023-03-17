import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.sparse as dglsp

class GATConv(nn.Module):
    def __init__(self, in_size, out_size, num_heads, dropout):
        super().__init__()

        self.out_size = out_size
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.W = nn.Linear(in_size, out_size * num_heads)
        self.a_l = nn.Parameter(torch.zeros(1, out_size, num_heads))
        self.a_r = nn.Parameter(torch.zeros(1, out_size, num_heads))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.W.weight, gain=gain)
        nn.init.xavier_normal_(self.a_l, gain=gain)
        nn.init.xavier_normal_(self.a_r, gain=gain)

    ###########################################################################
    # (HIGHLIGHT) Take the advantage of DGL sparse APIs to implement
    # multihead attention.
    ###########################################################################
    def forward(self, A_hat: dglsp.SparseMatrix, Z: torch.Tensor):
        Z = self.dropout(Z)
        Z = self.W(Z).view(Z.shape[0], self.out_size, self.num_heads)

        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        e_l = (Z * self.a_l).sum(dim=1)
        e_r = (Z * self.a_r).sum(dim=1)
        e = e_l[A_hat.row] + e_r[A_hat.col]

        a = F.leaky_relu(e)
        A_atten = dglsp.softmax(dglsp.val_like(A_hat, a))
        a_drop = self.dropout(A_atten.val)
        A_atten = dglsp.val_like(A_atten, a_drop)
        return dglsp.bspmm(A_atten, Z)

model = GATConv(10, 20, 8, 0.1)
scripted_model = torch.jit.script(model)
print(scripted_model.code)
