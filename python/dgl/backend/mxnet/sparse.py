import mxnet as mx
import numpy as np
from mxnet import nd
from ...sparse import _gspmm, _gsddmm
from ...base import dgl_warning, is_all, ALL
from .tensor import asnumpy, copy_to, zerocopy_from_numpy, context, to_backend_ctx

__all__ = ['gspmm', 'gsddmm', 'edge_softmax']


def _scatter_nd(index, src, n_rows):
    """Similar to PyTorch's scatter nd on first dimension."""
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
    """Similar to PyTorch's gather nd on first dimension."""
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


def _need_reduce_last_dim(ufeat, efeat):
    """Indicates whether to reduce the last dimension on edges
    in the backward pass of spmm,
    if so, use dot instead of mul."""
    ushp = ufeat.shape
    eshp = efeat.shape
    return ushp[1:-1] == eshp[1:-1] and eshp[-1] == 1 and ushp[-1] > 1


def _muldiv(op, x):
    return 1. / x if op == 'div' else x


def _addsub(op, x):
    return -x if op == 'sub' else x


def _expand(x, shape):
    return x.broadcast_to((x.shape[0], *shape))


class GSpMM(mx.autograd.Function):
    def __init__(self, gidx, op, reduce_op):
        super(GSpMM, self).__init__()
        self.gidx = gidx
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
                        _muldiv(op, _gather_nd(argY, _expand(Y, dZ.shape[1:]))) * dZ,
                        X.shape[0])
                elif op in ['add', 'sub', 'copy_lhs']:
                    dX = _scatter_nd(argX, dZ, X.shape[0])
            dX = _reduce_grad(dX, X.shape)
        else:
            dX = nd.zeros_like(X)
        if op != 'copy_lhs':
            if reduce_op == 'sum':
                if op == 'mul' and _need_reduce_last_dim(X, Y):
                    dY = _gsddmm(gidx, 'dot', X, dZ)
                elif op in ['mul', 'div']:
                    dY = _gsddmm(gidx, 'mul', X, dZ)
                    if op == 'div':
                        dY = -dY / (Y ** 2)
                elif op in ['add', 'sub', 'copy_rhs']:
                    dY = _gsddmm(gidx, 'copy_rhs', X, _addsub(op, dZ))
            else:
                if op in ['mul',  'div']:
                    dY = _scatter_nd(
                        argY,
                        _gather_nd(argX, _expand(X, dZ.shape[1:])) * dZ,
                        Y.shape[0])
                    if op == 'div':
                        dY = -dY / (Y ** 2)
                elif op in ['add', 'sub', 'copy_rhs']:
                    dY = _scatter_nd(argY, _addsub(op, dZ), Y.shape[0])
            dY = _reduce_grad(dY, Y.shape)
        else:
            dY = nd.zeros_like(Y)
        self.saved_tensors = None
        return dX, dY


def gspmm(gidx, op, reduce_op, lhs_data, rhs_data):
    func = GSpMM(gidx, op, reduce_op)
    ctx = to_backend_ctx(gidx.ctx)
    # XXX(minjie): There is a bug in MXNet's autograd system when one of the inputs
    #   does not require gradient. Although it still invokes the backward function,
    #   it does not set the gradient value to the correct buffer, resulting all the
    #   input gradients to be zero. Fix this by enforcing all the inputs to require
    #   gradients.
    if lhs_data is None:
        lhs_data = nd.zeros((1,), ctx=ctx)
        lhs_data.attach_grad()
    if rhs_data is None:
        rhs_data = nd.zeros((1,), ctx=ctx)
        rhs_data.attach_grad()
    return func(lhs_data, rhs_data)


class GSDDMM(mx.autograd.Function):
    def __init__(self, gidx, op, lhs_target, rhs_target):
        super(GSDDMM, self).__init__()
        self.gidx = gidx
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
        else:
            dX = nd.zeros_like(X)
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
                    if op == 'div':
                        dY = -dY / (Y ** 2)
            else:
                if op in ['add', 'sub', 'copy_rhs']:
                    dY = _addsub(op, dZ)
                else:  # mul, div, dot
                    dY = _gsddmm(gidx, 'mul', dZ, X, 'e', lhs_target)
                    if op == 'div':
                        dY = -dY / (Y ** 2)
            dY = _reduce_grad(dY, Y.shape)
        else:
            dY = nd.zeros_like(Y)
        self.saved_tensors = None
        return dX, dY


def gsddmm(gidx, op, lhs_data, rhs_data, lhs_target='u', rhs_target='v'):
    func = GSDDMM(gidx, op, lhs_target, rhs_target)
    ctx = to_backend_ctx(gidx.ctx)
    if lhs_data is None:
        lhs_data = nd.zeros((1,), ctx=ctx)
    if rhs_data is None:
        rhs_data = nd.zeros((1,), ctx=ctx)
    return func(lhs_data, rhs_data)


class EdgeSoftmax(mx.autograd.Function):
    def __init__(self, gidx, eids, norm_by):
        super(EdgeSoftmax, self).__init__()
        if not is_all(eids):
            gidx = gidx.edge_subgraph(eids.astype(gidx.dtype), True)
        if norm_by == 'src':
            gidx = gidx.reverse()
        self.gidx = gidx

    def forward(self, score):
        """Forward function.

        Pseudo-code:

        .. code:: python

            score = dgl.EData(g, score)
            score_max = score.dst_max()  # of type dgl.NData
            score = score - score_max  # edge_sub_dst, ret dgl.EData
            score_sum = score.dst_sum()  # of type dgl.NData
            out = score / score_sum    # edge_div_dst, ret dgl.EData
            return out.data
        """
        gidx = self.gidx
        score_max = _gspmm(gidx, 'copy_rhs', 'max', None, score)[0]
        score = mx.nd.exp(_gsddmm(gidx, 'sub', score, score_max, 'e', 'v'))
        score_sum = _gspmm(gidx, 'copy_rhs', 'sum', None, score)[0]
        out = _gsddmm(gidx, 'div', score, score_sum, 'e', 'v')
        self.save_for_backward(out)
        return out

    def backward(self, grad_out):
        """Backward function.

        Pseudo-code:

        .. code:: python

            g, out = ctx.backward_cache
            grad_out = dgl.EData(g, grad_out)
            out = dgl.EData(g, out)
            sds = out * grad_out  # type dgl.EData
            sds_sum = sds.dst_sum()  # type dgl.NData
            grad_score = sds - sds * sds_sum  # multiple expressions
        """
        out, = self.saved_tensors
        gidx = self.gidx
        sds = out * grad_out
        accum = gspmm(gidx, 'copy_rhs', 'sum', None, sds)
        grad_score = sds - gsddmm(gidx, 'mul', out, accum, 'e', 'v')
        self.save_tensors = None
        return grad_score


def edge_softmax(gidx, logits, eids=ALL, norm_by='dst'):
    softmax_op = EdgeSoftmax(gidx, eids, norm_by)
    return softmax_op(logits)
