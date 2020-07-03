import torch as th
from ...sparse import _gspmm, _gsddmm


class GSpMM(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, op, reduce_op, lhs_data, rhs_data):
        out = _gspmm(g, op, reduce_op, lhs_data, rhs_data)
        return out[0]

    @staticmethod
    def backward(ctx, grad):
        return None, None, None, None, None


class GSDDMM(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, op, lhs_data, rhs_data, lhs_target, rhs_target):
        out = _gsddmm(g, op, lhs_data, rhs_data, lhs_target, rhs_target)
        return out

    @staticmethod
    def backward(ctx, grad):
        return None, None, None, None, None, None


def gspmm(g, op, reduce_op, lhs_data, rhs_data):
    return GSpMM.apply(g, op, reduce_op, lhs_data, rhs_data)

def gsddmm(g, op, lhs_data, rhs_data, lhs_target='u', rhs_target='v'):
    return GSDDMM.apply(g, op, lhs_data, rhs_data, lhs_target, rhs_target)
