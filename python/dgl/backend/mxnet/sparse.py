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
    reduce_idx = np.nonzero(np.array(grad_shape) - np.array(in_shape))
    reduce_idx += 1  # skip batch dim
    if len(reduce_idx) > 0:
        grad = grad.sum(axis=tuple(reduce_idx.tolist()), keepdim=True)
    return grad.reshape(-1, *shape[1:])

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
            g_rev = gidx.reverse()
            if self.reduce_op == 'sum':
                if self.op in ['mul', 'div']:
                    dX = _gspmm(g_rev, '*', 'sum', dZ, _muldiv(self.op, Y))[0]
                elif self.op in ['add', 'sub']:
                    dX = _gspmm(g_rev, 'copy_lhs', 'sum', dZ, Y)[0]
                elif self.op == 'copy_u':
                    dX = _gspmm(g_rev, 'copy_u', 'sum', dZ, None)[0]
            else:
                #dX = nd.zeros((X.shape[0],) + dZ.shape[1:], dtype=X.dtype, ctx=X.ctx)
                if self.op in ['mul', 'div']:
                    dX = nd.scatter_nd(_muldiv(self.op, nd.gather(Y.broadcast_to(-1, *dZ.shape[1:]), argY)) * dZ,
                                       argX, (X.shape[0],) + dZ.shape[1:])
                elif self.op in ['add', 'sub', 'copy_u']:
                    dX = nd.scatter_nd(dZ, argX.long(), (X.shape[0],) + dZ.shape[1:])
            dX = _reduce_grad(dX, X.shape)
        if self.op != 'copy_u' and Y.grad is not None:
            if self.reduce_op == 'sum':
                if self.op in ['mul', 'div']:
                    dY = _gsddmm(gidx, '*', X, dZ)
                    if self.op == 'div': dY = -dY / (Y ** 2)
                elif self.op in ['add', 'sub', 'copy_e']:
                    dY = _gsddmm(gidx, 'copy_rhs', X, _addsub(self.op, dZ))
            else:
                #dY = nd.zeros((Y.shape[0],) + dZ.shape[1:], dtype=Y.dtype, ctx=Y.ctx)
                if self.op in ['mul',  'div']:
                    dY = nd.scatter_nd(nd.gather_nd(X.broadcast_to(-1, *dZ.shape[1:]), argX) * dZ,
                                       argY, (Y.shape[0],) + dZ.shape[1:])
                    if self.op == 'div': dY = -dY / (Y ** 2)
                elif self.op in ['add', 'sub', 'copy_e']:
                    dY = nd.scatter_nd(_addsub(self.op, dZ), argY)
            dY = _reduce_grad(dY, Y.shape)
        self.saved_tensors = None
        return dX, dY

def gspmm(g, op, reduce_op, lhs_data, rhs_data):
    func = GSpMM(g, op, reduce_op)
    return func(lhs_data, rhs_data)
