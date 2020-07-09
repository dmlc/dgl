import mxnet as mx
import numpy as np
from mxnet import nd
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
    # pad in_shape
    in_shape = (1,) * num_to_squeeze + in_shape
    reduce_idx = np.nonzero(np.asarray(grad_shape) - np.asarray(in_shape))[0]
    reduce_idx += 1  # skip batch dim
    grad = grad.sum(axis=tuple(reduce_idx), keepdims=True)
    return grad.reshape(shape)

def _muldiv(op, x):
    return 1. / x if op == 'div' else x

def _addsub(op, x):
    return -x if op == 'sub' else x

class GSpMM(mx.autograd.Function):
    def __init__(self, g, op, reduce_op):
        super(GSpMM, self).__init__()
        self.gidx = g._graph
        self.op = op
        self.reduce_op = reduce_op

    def forward(self, X, Y):
        out, (argX, argY) = _gspmm(self.gidx, self.op, self.reduce_op, X, Y)
        self.save_for_backward(X, Y, out, argX, argY)
        return out

    def backward(self, dZ):
        X, Y, out, argX, argY = self.saved_tensors
        dX, dY = None, None
        if self.op != 'copy_e' and X.grad is not None:
            g_rev = self.gidx.reverse()
            if self.reduce_op == 'sum':
                if self.op in ['mul', 'div']:
                    dX = _gspmm(g_rev, '*', 'sum', dZ, _muldiv(self.op, Y))[0]
                elif self.op in ['add', 'sub']:
                    dX = _gspmm(g_rev, 'copy_lhs', 'sum', dZ, Y)[0]
                elif self.op == 'copy_u':
                    dX = _gspmm(g_rev, 'copy_u', 'sum', dZ, None)[0]
            else:
                if self.op in ['mul', 'div']:
                    dX = nd.scatter_nd(_muldiv(self.op, nd.gather_nd(Y.broadcast_to(-1, *dZ.shape[1:]), argY)) * dZ,
                                       argX, (X.shape[0],) + dZ.shape[1:])
                elif self.op in ['add', 'sub', 'copy_u']:
                    dX = nd.scatter_nd(dZ, argX, (X.shape[0],) + dZ.shape[1:])
            dX = _reduce_grad(dX, X.shape)
        if self.op != 'copy_u' and Y.grad is not None:
            if self.reduce_op == 'sum':
                if self.op in ['mul', 'div']:
                    dY = _gsddmm(self.gidx, '*', X, dZ)
                    if self.op == 'div': dY = -dY / (Y ** 2)
                elif self.op in ['add', 'sub', 'copy_e']:
                    dY = _gsddmm(self.gidx, 'copy_rhs', X, _addsub(self.op, dZ))
            else:
                if self.op in ['mul',  'div']:
                    dY = nd.scatter_nd(nd.gather_nd(X.broadcast_to(-1, *dZ.shape[1:]), argX) * dZ,
                                       argY, (Y.shape[0],) + dZ.shape[1:])
                    if self.op == 'div': dY = -dY / (Y ** 2)
                elif self.op in ['add', 'sub', 'copy_e']:
                    dY = nd.scatter_nd(_addsub(self.op, dZ), argY)
            dY = _reduce_grad(dY, Y.shape)
        self.saved_tensors = None
        if dX is None:
            dX = nd.empty(())
        if dY is None:
            dY = nd.empty(())
        return dX, dY

def gspmm(g, op, reduce_op, lhs_data, rhs_data):
    func = GSpMM(g, op, reduce_op)
    if lhs_data is None:
        lhs_data = nd.empty(())
    if rhs_data is None:
        rhs_data = nd.empty(())
    return func(lhs_data, rhs_data)

class GSDDMM(mx.autograd.Function):
    def __init__(self, g, op, lhs_target, rhs_target):
        super(GSDDMM, self).__init__()
        self.gidx = g._graph
        self.op = op
        self.lhs_target = lhs_target
        self.rhs_target = rhs_target

    def forward(self, X, Y):
        out = _gsddmm(self.gidx, self.op, X, Y, self.lhs_target, self.rhs_target)
        self.save_for_backward(X, Y, out)
        return out

    def backward(self, dZ):
        X, Y, out = self.saved_tensors
        gidx, op = self.gidx, self.op
        lhs_target, rhs_target = self.lhs_target, self.rhs_target
        dX, dY = None, None
        if op != 'copy_rhs':
            if lhs_target in ['u', 'v']:
                _gidx = gidx if self.lhs_target == 'v' else gidx.reverse()
                if op in ['add', 'sub', 'copy_lhs']:
                    dX = _gspmm(_gidx, 'copy_e', 'sum', None, dZ)[0]
                else:  # mul, div, dot
                    if rhs_target == lhs_target:
                        dX = _gspmm(_gidx, 'copy_e', 'sum', None, dZ)[0] * _muldiv(op, Y)
                    elif self.rhs_target == 'e':
                        dX = _gspmm(_gidx, 'copy_e', 'sum', None, dZ * _muldiv(op, Y))[0]
                    else:  # rhs_target = !lhs_target
                        dX = _gspmm(_gidx, 'mul', 'sum', _muldiv(op, Y), dZ)[0]
            else:  # lhs_target == 'e'
                if op in ['add', 'sub', 'copy_lhs']:
                    dX = dZ
                else:  # mul, div, dot
                    dX = _gsddmm(gidx, 'mul', dZ, _muldiv(op, Y), 'e', rhs_target)
            dX = _reduce_grad(dX, X.shape)
        if op != 'copy_lhs':
            if self.rhs_target in ['u', 'v']:
                _gidx = gidx if rhs_target == 'v' else gidx.reverse()
                if op in ['add', 'sub', 'copy_rhs']:
                    dY = _gspmm(_gidx, 'copy_e', 'sum', None, _addsub(op, dZ))[0]
                else:  # mul, div, dot
                    if lhs_target == rhs_target:
                        dY = _gspmm(_gidx, 'copy_e', 'sum', None, dZ)[0] * X
                    elif self.lhs_target == 'e':
                        dY = _gspmm(_gidx, 'copy_e', 'sum', None, dZ * X)[0]
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
        if dX is None:
            dX = nd.empty(())
        if dY is None:
            dY = nd.empty(())
        return dX, dY

def gsddmm(g, op, lhs_data, rhs_data, lhs_target='u', rhs_target='v'):
    func = GSDDMM(g, op, lhs_target, rhs_target)
    if lhs_data is None:
        lhs_data = nd.empty(())
    if rhs_data is None:
        rhs_data = nd.empty(())
    return func(lhs_data, rhs_data)
