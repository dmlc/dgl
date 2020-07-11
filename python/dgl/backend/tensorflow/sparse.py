import tensorflow as tf
import numpy as np
from .tensor import tensor, copy_to, context
from ...sparse import _gspmm, _gsddmm

def _scatter_nd(index, src, n_rows):
    assert index.shape == src.shape
    shp = index.shape
    ctx = context(src)
    ndim = index.ndim
    offsets = []
    stride = 1
    for i in reversed(range(1, ndim)):
        di = shp[i]
        offset_i = tf.range(di, dtype=index.dtype)
        offsets.append(
            tf.reshape((stride * offset_i), (1,) * i + (di,) + (1,) * (ndim - 1 - i)))
        stride *= di
    new_idx = index * stride + copy_to(sum(offsets), ctx)
    src = tf.reshape(src, (-1,))
    new_idx = tf.reshape(new_idx, (-1, 1))
    rst = tf.reshape(tf.scatter_nd(new_idx, src, (stride * n_rows,)), (n_rows, *shp[1:]))
    return rst

def _gather_nd(index, src):
    shp = index.shape
    ctx = context(src)
    ndim = index.ndim
    offsets = []
    stride = 1
    for i in reversed(range(1, ndim)):
        di = shp[i]
        offset_i = tf.range(di, dtype=index.dtype)
        offsets.append(
            tf.reshape((stride * offset_i), (1,) * i + (di,) + (1,) * (ndim - 1 - i)))
        stride *= di
    new_idx = index * stride + copy_to(sum(offsets), ctx)
    src = tf.reshape(src, (-1,))
    new_idx = tf.reshape(new_idx, (-1))
    print(src, new_idx)
    rst = tf.reshape(tf.gather(src, new_idx), shp)
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
    reduce_idx = np.asarray(np.nonzero(np.asarray(grad_shape) - np.asarray(in_shape)))
    reduce_idx += 1  # skip batch dim
    reduce_idx_tensor = tf.constant(tuple(
        reduce_idx.flatten().tolist()))
    grad = tf.reduce_sum(grad, axis=reduce_idx_tensor, keepdims=True)
    return tf.reshape(grad, shape)

def _muldiv(op, x):
    return 1. / x if op == 'div' else x

def _addsub(op, x):
    return -x if op == 'sub' else x

def gspmm_real(g, op, reduce_op, X, Y):
    gidx = g._graph
    out, (argX, argY) = _gspmm(gidx, op, reduce_op, X, Y)

    def grad(dZ):
        dZ = tensor(dZ)
        dX, dY = tf.zeros(()), tf.zeros(())
        if op != 'copy_rhs':
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
                        _muldiv(op, _gather_nd(argY, tf.broadcast_to(Y, (Y.shape[0], *dZ.shape[1:])))) * dZ,
                        X.shape[0])
                elif op in ['add', 'sub', 'copy_lhs']:
                    dX = _scatter_nd(argX, dZ, X.shape[0])
            dX = _reduce_grad(dX, X.shape)
        if op != 'copy_lhs':
            if reduce_op == 'sum':
                if op in ['mul', 'div']:
                    dY = _gsddmm(gidx, 'mul', X, dZ)
                    if op == 'div': dY = -dY / (Y ** 2)
                elif op in ['add', 'sub', 'copy_rhs']:
                    dY = _gsddmm(gidx, 'copy_rhs', X, _addsub(op, dZ))
            else:
                out_shp = (Y.shape[0],) + dZ.shape[1:]
                if op in ['mul',  'div']:
                    dY = _scatter_nd(
                        argY,
                        _gather_nd(argX, tf.broadcast_to(X, (X.shape[0], *dZ.shape[1:]))) * dZ,
                        Y.shape[0])
                    if op == 'div': dY = -dY / (Y ** 2)
                elif op in ['add', 'sub', 'copy_rhs']:
                    dY = _scatter_nd(argY, _addsub(op, dZ), Y.shape[0])
            dY = _reduce_grad(dY, Y.shape)
        return dX, dY
    return out, grad

def gspmm(g, op, reduce_op, X, Y):
    @tf.custom_gradient
    def _lambda(X, Y):
        return gspmm_real(g, op, reduce_op, X, Y)
    return _lambda(X, Y)

def gsddmm_real(g, op, X, Y, lhs_target, rhs_target):
    gidx = g._graph
    out = _gsddmm(gidx, op, X, Y, lhs_target, rhs_target)

    def grad(dZ):
        dX, dY = tf.zeros(()), tf.zeros(())
        if op != 'copy_rhs':
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
        if op != 'copy_lhs':
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
        return dX, dY
    return out, grad

def gsddmm(g, op, X, Y, lhs_target='u', rhs_target='v'):
    @tf.custom_gradient
    def _lambda(X, Y):
        return gsddmm_real(g, op, X, Y, lhs_target, rhs_target)
    return _lambda(X, Y)
