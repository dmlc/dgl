"""Various commonly used linear modules"""
# pylint: disable= no-member, arguments-differ, invalid-name, W0235
import torch
import torch.nn as nn

from ...ops import segment_mm, gather_mm

__all__ = ['TypedLinear']

class TypedLinear(nn.Module):
    """Linear transformation according to types.

    Given a batch of input samples :math:`X` and :math:`W` 

    Parameters
    ----------
    """
    def __init__(self, in_size, out_size, num_types,
                 regularizer=None, num_bases=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_types = num_types
        if regularizer is None:
            self.W = nn.Parameter(torch.Tensor(num_types, in_size, out_size))
        elif regularizer == 'basis':
            if num_bases is None:
                raise ValueError('Missing "num_bases" for basis regularization.')
            self.W = nn.Parameter(torch.Tensor(num_bases, in_size, out_size))
            self.coeff = nn.Parameter(torch.Tensor(num_types, num_bases))
            self.num_bases = num_bases
        elif regularizer == 'bdd':
            if in_size % num_bases != 0 or out_size % num_bases != 0:
                raise ValueError(
                    'Input and output sizes must be a multiplier of num_bases.'
                )
            self.submat_in = in_size // num_bases
            self.submat_out = out_size // num_bases
            self.W = nn.Parameter(torch.Tensor(
                num_types, num_bases * self.submat_in * self.submat_out))
            self.num_bases = num_bases
        else:
            raise ValueError(
                f'Supported regularizer options: "basis", "bdd", but got {regularizer}')
        self.regularizer = regularizer
        self.reset_parameters()

    def reset_parameters(self):
        if self.regularizer is None:
            nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
        elif self.regularizer == 'basis':
            nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.coeff, gain=nn.init.calculate_gain('relu'))
        elif self.regularizer == 'bdd':
            nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
        else:
            raise ValueError(
                f'Supported regularizer options: "basis", "bdd", but got {regularizer}')

    def get_weight(self):
        if self.regularizer is None:
            return self.W
        elif self.regularizer == 'basis':
            W = self.W.view(self.num_bases, self.in_size * self.out_size)
            return (self.coeff @ W).view(self.num_types, self.in_size, self.out_size)
        elif self.regularizer == 'bdd':
            return self.W
        else:
            raise ValueError(
                f'Supported regularizer options: "basis", "bdd", but got {regularizer}')

    def forward(self, x, x_type, sorted_by_type=False):
        w = self.get_weight()
        if self.regularizer == 'bdd':
            w = w.index_select(0, x_type).view(-1, self.submat_in, self.submat_out)
            x = x.view(-1, 1, self.submat_in)
            return torch.bmm(x, w).view(-1, self.out_size)
        elif sorted_by_type:
            pos_l = torch.searchsorted(x_type, torch.arange(self.num_types, device=x.device))
            pos_r = torch.cat([pos_l[1:], torch.tensor([len(x_type)], device=x.device)])
            seglen = (pos_r - pos_l).cpu()  # XXX(minjie): cause device synchronize
            return segment_mm(x, w, seglen_a=seglen)
        else:
            return gather_mm(x, w, idx_b=x_type)

    def __repr__(self):
        if self.regularizer is None:
            return (f'TypedLinear(in_size={self.in_size}, out_size={self.out_size}, '
                    f'num_types={self.num_types})')
        else:
            return (f'TypedLinear(in_size={self.in_size}, out_size={self.out_size}, '
                    f'num_types={self.num_types}, regularizer={self.regularizer}), '
                    f'num_bases={self.num_bases}')
