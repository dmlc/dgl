import mxnet as mx
import numpy as np
from mxnet import nd
from ...sparse import _gspmm, _gsddmm
from ...base import dgl_warning
from .tensor import asnumpy, copy_to, zerocopy_from_numpy, context

def _scatter_nd(index, src, n_rows):
    assert index.shape == src.shape
    dgl_warning("MXNet do not support scatter_add, fallback to numpy.")
    ctx = context(src)
    index = asnumpy(index)
    src = asnumpy(src)
    shp = index.shape
    ndim = src.ndim
    offsets = []
    stride = 1
    for i in reversed(range(1, ndim)):
        di = shp[i]
        offset_i = np.arange(di, dtype=index.dtype)
        offsets.append(
            (stride * offset_i).reshape((1,) * i + (di,) + (1,) * (ndim - 1 - i)))
        stride *= di
    new_idx = index * stride + sum(offsets)
    src = src.reshape(-1)
    new_idx = new_idx.reshape(-1)
    rst = np.zeros((stride * n_rows,), dtype=src.dtype)
    np.add.at(rst, new_idx, src)
    rst = rst.reshape(n_rows, *shp[1:])
    rst = copy_to(zerocopy_from_numpy(rst), ctx)
    return rst

def _gather_nd(index, src):
    ctx = context(src)
    shp = index.shape
    ndim = src.ndim
    offsets = []
    stride = 1
    for i in reversed(range(1, ndim)):
        di = shp[i]
        offset_i = nd.arange(di, dtype=index.dtype)
        offsets.append(
            (stride * offset_i).reshape((1,) * i + (di,) + (1,) * (ndim - 1 - i)))
        stride *= di
    new_idx = index * stride + copy_to(sum(offsets), ctx)
    src = src.reshape(-1)
    new_idx = new_idx.reshape(-1)
    rst = nd.take(src, new_idx).reshape(shp)
    return rst

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
        self.save_for_backward(X, Y, argX, argY)
        return out

    def backward(self, dZ):
        ctx = context(dZ)
        X, Y, argX, argY = self.saved_tensors
        gidx, op, reduce_op = self.gidx, self.op, self.reduce_op
        dX, dY = nd.empty((), ctx=ctx), nd.empty((), ctx=ctx)
        if op != 'copy_rhs' and X.grad is not None:
            g_rev = gidx.reverse()
            if reduce_op == 'sum':
                if op in ['mul', 'div']:
                    dX = _gspmm(g_rev, 'mul', 'sum', dZ, _muldiv(op, Y))[0]
                elif op in ['add', 'sub']:
                    dX = _gspmm(g_rev, 'copy_lhs', 'sum', dZ, Y)[0]
                elif op == 'copy_lhs':
                    dX = _gspmm(g_rev, 'copy_lhs', 'sum', dZ, None)[0]
            else:
                if op in ['mul', 'div']:
                    dX = _scatter_nd(
                        argX,
                        _muldiv(op, _gather_nd(argY, Y.broadcast_to((Y.shape[0], *dZ.shape[1:])))) * dZ,
                        X.shape[0])
                elif op in ['add', 'sub', 'copy_lhs']:
                    dX = _scatter_nd(argX, dZ, X.shape[0])
            dX = _reduce_grad(dX, X.shape)
        if op != 'copy_lhs' and Y.grad is not None:
            if reduce_op == 'sum':
                if op in ['mul', 'div']:
                    dY = _gsddmm(gidx, 'mul', X, dZ)
                    if op == 'div': dY = -dY / (Y ** 2)
                elif op in ['add', 'sub', 'copy_rhs']:
                    dY = _gsddmm(gidx, 'copy_rhs', X, _addsub(op, dZ))
            else:
                if op in ['mul',  'div']:
                    dY = _scatter_nd(
                        argY,
                        _gather_nd(argX, X.broadcast_to((X.shape[0], *dZ.shape[1:]))) * dZ,
                        Y.shape[0])
                    if op == 'div': dY = -dY / (Y ** 2)
                elif op in ['add', 'sub', 'copy_rhs']:
                    dY = _scatter_nd(argY, _addsub(op, dZ), Y.shape[0])
            dY = _reduce_grad(dY, Y.shape)
        self.saved_tensors = None
        return dX, dY

def gspmm(g, op, reduce_op, lhs_data, rhs_data):
    func = GSpMM(g, op, reduce_op)
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
        self.save_for_backward(X, Y)
        return out

    def backward(self, dZ):
        ctx = context(dZ)
        X, Y = self.saved_tensors
        gidx, op = self.gidx, self.op
        lhs_target, rhs_target = self.lhs_target, self.rhs_target
        dX, dY = nd.empty((), ctx=ctx), nd.empty((), ctx=ctx)
        if op != 'copy_rhs':
            if lhs_target in ['u', 'v']:
                _gidx = gidx if self.lhs_target == 'v' else gidx.reverse()
                if op in ['add', 'sub', 'copy_lhs']:
                    dX = _gspmm(_gidx, 'copy_rhs', 'sum', None, dZ)[0]
                else:  # mul, div, dot
                    if rhs_target == lhs_target:
                        dX = _gspmm(_gidx, 'copy_rhs', 'sum', None, dZ)[0] * _muldiv(op, Y)
                    elif self.rhs_target == 'e':
                        dX = _gspmm(_gidx, 'copy_rhs', 'sum', None, dZ * _muldiv(op, Y))[0]
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
                    dY = _gspmm(_gidx, 'copy_rhs', 'sum', None, _addsub(op, dZ))[0]
                else:  # mul, div, dot
                    if lhs_target == rhs_target:
                        dY = _gspmm(_gidx, 'copy_rhs', 'sum', None, dZ)[0] * X
                    elif self.lhs_target == 'e':
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
        self.saved_tensors = None
        return dX, dY

def gsddmm(g, op, lhs_data, rhs_data, lhs_target='u', rhs_target='v'):
    func = GSDDMM(g, op, lhs_target, rhs_target)
    return func(lhs_data, rhs_data)
