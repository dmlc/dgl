import torch as th
import torch.nn as nn
import torch.nn.functional as F

'''
Defult modules: this is Pytorch specific
    - MessageModule: copy
    - UpdateModule: vanilla RNN
    - ReadoutModule: bag of words
    - ReductionModule: bag of words
'''

class DefaultMessageModule(nn.Module):
    """
    Default message module:
        - copy
    """
    def __init__(self, *args, **kwargs):
        super(DefaultMessageModule, self).__init__(*args, **kwargs)

    def forward(self, x):
        return x

class DefaultUpdateModule(nn.Module):
    """
    Default update module:
        - a vanilla GRU with ReLU, or GRU
    """
    def __init__(self, *args, **kwargs):
        super(DefaultUpdateModule, self).__init__()
        h_dims = self.h_dims = kwargs.get('h_dims', 128)
        net_type = self.net_type = kwargs.get('net_type', 'fwd')
        n_func = self.n_func = kwargs.get('n_func', 1)
        self.f_idx = 0
        self.reduce_func = DefaultReductionModule()
        if net_type == 'gru':
            self.net = [nn.GRUCell(h_dims, h_dims) for i in range(n_func)]
        else:
            self.net = [nn.Linear(2 * h_dims, h_dims) for i in range(n_func)]

    def forward(self, x, msgs):
        if not th.is_tensor(x):
            x = th.zeros_like(msgs[0])
        m = self.reduce_func(msgs)
        assert(self.f_idx < self.n_func)
        if self.net_type == 'gru':
            out = self.net[self.f_idx](m, x)
        else:
            _in = th.cat((m, x), 1)
            out = F.relu(self.net[self.f_idx](_in))
        self.f_idx += 1
        return out

    def reset_f_idx(self):
        self.f_idx = 0

class DefaultReductionModule(nn.Module):
    """
    Default readout:
        - bag of words
    """
    def __init__(self, *args, **kwargs):
        super(DefaultReductionModule, self).__init__(*args, **kwargs)

    def forward(self, x_s):
        out = th.stack(x_s)
        out = th.sum(out, dim=0)
        return out

class DefaultReadoutModule(nn.Module):
    """
    Default readout:
        - bag of words
    """
    def __init__(self, *args, **kwargs):
        super(DefaultReadoutModule, self).__init__(*args, **kwargs)
        self.reduce_func = DefaultReductionModule()

    def forward(self, x_s):
        return self.reduce_func(x_s)

