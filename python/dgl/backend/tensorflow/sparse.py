import tensorflow as tf
import numpy as np
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

def gspmm_real(g, op, reduce_op, X, Y):
    gidx = g._graph
    out, (argX, argY) = _gspmm(gidx, op, reduce_op, X, Y)

    def grad(dZ):
        dX, dY = None, None
        if op != 'copy_e' and ctx.needs_input_grad[3]:
            g_rev = gidx.reverse()
            if reduce_op == 'sum':
                if op in ['mul', 'div']:
                    dX = _gspmm(g_rev, '*', 'sum', dZ, _muldiv(op, Y))[0]
                elif op in ['add', 'sub']:
                    dX = _gspmm(g_rev, 'copy_lhs', 'sum', dZ, Y)[0]
                elif op == 'copy_u':
                    dX = _gspmm(g_rev, 'copy_u', 'sum', dZ, None)[0]
            else:
                dX = th.zeros((X.shape[0],) + dZ.shape[1:], dtype=X.dtype, device=X.device)
                if op in ['mul', 'div']:
                    dX.scatter_add_(0, argX.long(),
                                    _muldiv(op, Y.expand(-1, *dZ.shape[1:]).gather(0, argY.long())) * dZ)
                elif op in ['add', 'sub', 'copy_u']:
                    dX.scatter_add_(0, argX.long(), dZ)
            dX = _reduce_grad(dX, X.shape)
        if op != 'copy_u' and ctx.needs_input_grad[4]:
            if reduce_op == 'sum':
                if op in ['mul', 'div']:
                    dY = _gsddmm(gidx, '*', X, dZ)
                    if op == 'div': dY = -dY / (Y ** 2)
                elif op in ['add', 'sub', 'copy_e']:
                    dY = _gsddmm(gidx, 'copy_rhs', X, _addsub(op, dZ))
            else:
                dY = th.zeros((Y.shape[0],) + dZ.shape[1:], dtype=Y.dtype, device=Y.device)
                if op in ['mul',  'div']:
                    dY.scatter_add_(0, argY.long(),
                                    X.expand(-1, *dZ.shape[1:]).gather(0, argX.long()) * dZ)
                    if op == 'div': dY = -dY / (Y ** 2)
                elif op in ['add', 'sub', 'copy_e']:
                    dY.scatter_add_(0, argY.long(), _addsub(op, dZ))
            dY = _reduce_grad(dY, Y.shape)
        return dX, dY

    return out, grad

def gspmm(g, op, reduce_op, X, Y):
    @tf.custom_gradient
    def _lambda(X, Y):
        return gspmm_real(g, op, reduce_op, X, Y)
    return _lambda(X, Y)

def gsddmm_real(g, op, X, Y, lhs_target, rhs_target):
    pass

def gsddmm(g, op, X, Y, lhs_target='u', rhs_target='v'):
    @tf.custom_gradient
    def _lambda(X, Y):
        return gsddmm_real(g, op, X, Y, lhs_target, rhs_target)
    return _lambda
