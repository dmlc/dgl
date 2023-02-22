import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from model.tensorized_layers.graphsage import BatchedGraphSAGE


class DiffPoolAssignment(nn.Module):
    def __init__(self, nfeat, nnext):
        super().__init__()
        self.assign_mat = BatchedGraphSAGE(nfeat, nnext, use_bn=True)

    def forward(self, x, adj, log=False):
        s_l_init = self.assign_mat(x, adj)
        s_l = F.softmax(s_l_init, dim=-1)
        return s_l
