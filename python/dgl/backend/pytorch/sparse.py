import torch as th
from ...sparse import _gspmm, _gsddmm

def _reduce_grad(grad, shape):
    """Reduce gradient on the broadcast dimension
    If there is broadcast in forward pass, gradients need to be reduced on
    broadcast dimension. This function checks the input tensor shape and
    gradient shape and perform the reduction.
    Parameters
    ----------
    grad: Tensor
        Gradient tensor
    shape: tuple
        Shape of input tensor
    Returns
    -------
    Tensor
    """
    grad_shape = grad.shape[1:]
    in_shape = shape[1:]
    if in_shape == grad_shape:
        # no need to reduce
        return grad
    num_to_squeeze = len(grad_shape) - len(in_shape)
    # pad inshape
    in_shape = (1,) * num_to_squeeze + in_shape
    reduce_idx = th.nonzero(th.tensor(grad_shape) - th.tensor(in_shape))
    reduce_idx += 1  # skip batch dim
    if len(reduce_idx) > 0:
        grad = grad.sum(dim=tuple(reduce_idx), keepdim=True)
    return grad.view(-1, *shape[1:])

class GSpMM(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, op, reduce_op, X, Y):
        gidx = g._graph
        out, (argX, argY) = _gspmm(gidx, op, reduce_op, X, Y)
        ctx.backward_cache = gidx, op, reduce_op
        ctx.save_for_backward(X, Y, out, argX, argY)
        return out

    @staticmethod
    def backward(ctx, dZ):
        gidx, op, reduce_op = ctx.backward_cache
        X, Y, out, argX, argY = ctx.saved_tensors
        dX, dY = None, None
        if ctx.needs_input_grad[3]:
            g_rev = gidx.reverse()
            if reduce_op == 'sum':
                if op == 'mul':
                    dX = _gspmm(g_rev, '*', 'sum', dZ, Y)
                elif op in ['add', 'sub']:
                    dX = _gspmm(g_rev, 'copy_lhs', 'sum', dZ, Y)
                elif op == 'div':
                    dX = _gspmm(g_rev, '*', 'sum', dZ, 1. / Y)
            dX = _reduce_grad(dX, X.shape)
        if ctx.needs_input_grad[4]:
            if reduce_op == 'sum':
                if op == 'mul':
                    dY = _gsddmm(gidx, '*', X, dZ)
                elif op == 'add':
                    dY = _gsddmm(gidx, 'copy_rhs', X, dZ)
                elif op == 'sub':
                    dY = _gsddmm(gidx, 'copy_rhs', X, -dZ)
                elif op == 'div':
                    dY = -_gsddmm(gidx, '*', X, dZ) / (Y ** 2)
            else:
                pass
            dY = _reduce_grad(dY, Y.shape)
        return None, None, None, dX, dY

class GSDDMM(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, op, lhs_data, rhs_data, lhs_target, rhs_target):
        gidx = g._graph
        out = _gsddmm(gidx, op, lhs_data, rhs_data, lhs_target, rhs_target)
        return out

    @staticmethod
    def backward(ctx, grad):
        return None, None, None, None, None, None

class EdgeSoftmax(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, score):
        gidx = g._graph
        score_max = _gspmm(gidx, 'copy_e', 'max', None, score)[0]
        score = th.exp(_gsddmm(gidx, '-', score, score_max, 'e', 'v'))
        score_sum = _gspmm(gidx, 'copy_e', 'sum', None, score)[0]
        out = _gsddmm(gidx, '/', score, score_sum, 'e', 'v')
        ctx.backward_cache = gidx
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad):
        gidx = ctx.backward_cache
        out, = ctx.saved_tensors
        sds = out * grad
        accum = _gspmm(gidx, 'copy_e', 'sum', None, sds)[0]
        out = _gsddmm(gidx, '*', out, accum, 'e', 'v')
        grad_score = sds - out
        return None, grad_score, None


def gspmm(g, op, reduce_op, lhs_data, rhs_data):
    return GSpMM.apply(g, op, reduce_op, lhs_data, rhs_data)

def gsddmm(g, op, lhs_data, rhs_data, lhs_target='u', rhs_target='v'):
    return GSDDMM.apply(g, op, lhs_data, rhs_data, lhs_target, rhs_target)

def edge_softmax(g, score):
    return EdgeSoftmax.apply(g, score)
