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


class EdgeSoftmax(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, score):
        score_max = _gspmm(g, 'copy_e', 'max', None, score)[0]
        score = th.exp(_gsddmm(g, '-', score, score_max, 'e', 'v'))
        score_sum = _gspmm(g, 'copy_e', 'sum', None, score)[0]
        out = _gsddmm(g, '/', score, score_sum, 'e', 'v')
        ctx.backward_cache = g
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad):
        g = ctx.backward_cache
        out, = ctx.saved_tensors
        sds = out * grad
        accum = _gspmm(g, 'copy_e', 'sum', None, sds)[0]
        out = _gsddmm(g, '*', out, accum, 'e', 'v')
        grad_score = sds - out
        return None, grad_score, None


def gspmm(g, op, reduce_op, lhs_data, rhs_data):
    return GSpMM.apply(g, op, reduce_op, lhs_data, rhs_data)

def gsddmm(g, op, lhs_data, rhs_data, lhs_target='u', rhs_target='v'):
    return GSDDMM.apply(g, op, lhs_data, rhs_data, lhs_target, rhs_target)

def edge_softmax(g, score):
    return EdgeSoftmax.apply(g, score)
