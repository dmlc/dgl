import torch as th
from ...sparse import _gspmm, _gsddmm

__all__ = ['gspmm', 'gsddmm']

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
        Shape of input tens
    or
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

def _muldiv(op, x):
    return 1. / x if op == 'div' else x

def _addsub(op, x):
    return -x if op == 'sub' else x

class GSpMM(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, op, reduce_op, X, Y):
        gidx = g._graph
        out, (argX, argY) = _gspmm(gidx, op, reduce_op, X, Y)
        ctx.backward_cache = gidx, op, reduce_op
        ctx.save_for_backward(X, Y, argX, argY)
        return out

    @staticmethod
    def backward(ctx, dZ):
        gidx, op, reduce_op = ctx.backward_cache
        X, Y, argX, argY = ctx.saved_tensors
        dX, dY = None, None
        if op != 'copy_rhs' and ctx.needs_input_grad[3]:
            g_rev = gidx.reverse()
            if reduce_op == 'sum':
                if op in ['mul', 'div']:
                    dX = _gspmm(g_rev, 'mul', 'sum', dZ, _muldiv(op, Y))[0]
                elif op in ['add', 'sub']:
                    dX = _gspmm(g_rev, 'copy_lhs', 'sum', dZ, Y)[0]
                elif op == 'copy_lhs':
                    dX = _gspmm(g_rev, 'copy_lhs', 'sum', dZ, None)[0]
            else:
                dX = th.zeros((X.shape[0],) + dZ.shape[1:], dtype=X.dtype, device=X.device)
                if op in ['mul', 'div']:
                    dX.scatter_add_(0, argX.long(),
                                    _muldiv(op, Y.expand(-1, *dZ.shape[1:]).gather(0, argY.long())) * dZ)
                elif op in ['add', 'sub', 'copy_lhs']:
                    dX.scatter_add_(0, argX.long(), dZ)
            dX = _reduce_grad(dX, X.shape)
        if op != 'copy_lhs' and ctx.needs_input_grad[4]:
            if reduce_op == 'sum':
                if op in ['mul', 'div']:
                    dY = _gsddmm(gidx, 'mul', X, dZ)
                    if op == 'div': dY = -dY / (Y ** 2)
                elif op in ['add', 'sub', 'copy_rhs']:
                    dY = _gsddmm(gidx, 'copy_rhs', X, _addsub(op, dZ))
            else:
                dY = th.zeros((Y.shape[0],) + dZ.shape[1:], dtype=Y.dtype, device=Y.device)
                if op in ['mul',  'div']:
                    dY.scatter_add_(0, argY.long(),
                                    X.expand(-1, *dZ.shape[1:]).gather(0, argX.long()) * dZ)
                    if op == 'div': dY = -dY / (Y ** 2)
                elif op in ['add', 'sub', 'copy_rhs']:
                    dY.scatter_add_(0, argY.long(), _addsub(op, dZ))
            dY = _reduce_grad(dY, Y.shape)
        return None, None, None, dX, dY

class GSDDMM(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, op, X, Y, lhs_target, rhs_target):
        gidx = g._graph
        out = _gsddmm(gidx, op, X, Y, lhs_target, rhs_target)
        ctx.backward_cache = gidx, op, lhs_target, rhs_target
        ctx.save_for_backward(X, Y)
        return out

    @staticmethod
    def backward(ctx, dZ):
        gidx, op, lhs_target, rhs_target = ctx.backward_cache
        X, Y = ctx.saved_tensors
        dX, dY = None, None
        if op != 'copy_rhs' and ctx.needs_input_grad[2]:
            if lhs_target in ['u', 'v']:
                _gidx = gidx if lhs_target == 'v' else gidx.reverse()
                if op in ['add', 'sub', 'copy_lhs']:
                    dX = _gspmm(_gidx, 'copy_rhs', 'sum', None, dZ)[0]
                else:  # mul, div, dot
                    if rhs_target == lhs_target:
                        dX = _gspmm(_gidx, 'copy_rhs', 'sum', None, dZ)[0] * _muldiv(op, Y)
                    elif rhs_target == 'e':
                        dX = _gspmm(_gidx, 'copy_rhs', 'sum', None, dZ * _muldiv(op, Y))[0]
                    else:  # rhs_target = !lhs_target
                        dX = _gspmm(_gidx, 'mul', 'sum', _muldiv(op, Y), dZ)[0]
            else:  # lhs_target == 'e'
                if op in ['add', 'sub', 'copy_lhs']:
                    dX = dZ
                else:  # mul, div, dot
                    dX = _gsddmm(gidx, 'mul', dZ, _muldiv(op, Y), 'e', rhs_target)
            dX = _reduce_grad(dX, X.shape)
        if op != 'copy_lhs' and ctx.needs_input_grad[3]:
            if rhs_target in ['u', 'v']:
                _gidx = gidx if rhs_target == 'v' else gidx.reverse()
                if op in ['add', 'sub', 'copy_rhs']:
                    dY = _gspmm(_gidx, 'copy_rhs', 'sum', None, _addsub(op, dZ))[0]
                else:  # mul, div, dot
                    if lhs_target == rhs_target:
                        dY = _gspmm(_gidx, 'copy_rhs', 'sum', None, dZ)[0] * X
                    elif lhs_target == 'e':
                        dY = _gspmm(_gidx, 'copy_rhs', 'sum', None, dZ * X)[0]
                    else:  # rhs_target = !lhs_target
                        dY = _gspmm(_gidx, 'mul', 'sum', X, dZ)[0]
                    if op == 'div': dY = -dY / (Y ** 2)
            else:
                if op in ['add', 'sub', 'copy_rhs']:
                    dY = _addsub(op, dZ)
                else:  # mul, div, dot
                    dY = _gsddmm(gidx, 'mul', dZ, X, 'e', lhs_target)
                    if op == 'div': dY = -dY / (Y ** 2)
            dY = _reduce_grad(dY, Y.shape)
        return None, None, dX, dY, None, None

def gspmm(g, op, reduce_op, lhs_data, rhs_data):
    return GSpMM.apply(g, op, reduce_op, lhs_data, rhs_data)

def gsddmm(g, op, lhs_data, rhs_data, lhs_target='u', rhs_target='v'):
    return GSDDMM.apply(g, op, lhs_data, rhs_data, lhs_target, rhs_target)

