import torch
import torch.nn as nn
from torch.autograd import Variable
import os

def create_var(tensor, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor)
    else:
        return Variable(tensor, requires_grad=requires_grad)

def cuda(tensor):
    if torch.cuda.is_available() and not os.getenv('NOCUDA', None):
        return tensor.cuda()
    else:
        return tensor


class GRUUpdate(nn.Module):
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size

        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, node):
        src_x = node.data['src_x']
        dst_x = node.data['dst_x']
        s = node.data['s']
        rm = node.data['accum_rm']
        z = torch.sigmoid(self.W_z(torch.cat([src_x, s], 1)))
        m = torch.tanh(self.W_h(torch.cat([src_x, rm], 1)))
        m = (1 - z) * s + z * m
        r_1 = self.W_r(dst_x)
        r_2 = self.U_r(m)
        r = torch.sigmoid(r_1 + r_2)

        return {'m': m, 'r': r, 'z': z, 'rm': r * m}

