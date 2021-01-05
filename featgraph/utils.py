""" Utilities used in GSpMM and GSDDMM operators. """
from tvm.tir import Select, const
from tvm import te


def reducible(x, y, ndim):
    x0 = x & 1
    y0 = y & 1
    if x0 == 1 and y0 == 1:
        return True
    x >>= 1
    y >>= 1
    for _ in range(ndim - 1):
        x1 = x & 1
        y1 = y & 1
        if x1 == y1 and x0 == y0:
            return True
        if x1 == 1 and y1 == 1:
            return True
        if x1 == 1 and x0 == 1:
            return True
        if y1 == 1 and y0 == 1:
            return True
        x0, y0 = x1, y1
        x >>= 1
        y >>= 1
    return False


def binary_to_code(b, ndim):
    rst = ""
    for _ in range(ndim):
        rst += "1" if b & 1 else "x"
        b >>= 1
    return rst


def irreducible_ops(ndim):
    ret = []
    for i in range(1 << ndim):
        for j in range(1 << ndim): 
            if not reducible(i, j, ndim):
                ret.append((binary_to_code(i, ndim), binary_to_code(j, ndim)))
    return ret


def max_combine(x, y):
    if len(x) == 3:
        eid = Select(x[2] > y[2], x[0], y[0])
        cid = Select(x[2] > y[2], x[1], y[1])
        val = Select(x[2] > y[2], x[2], y[2])
        return eid, cid, val
    else:
        idx = Select(x[1] > y[1], x[0], y[0])
        val = Select(x[1] > y[1], x[1], y[1])
        return idx, val


def max_identity(x, y, z=None):
    if z:
        return const(0, x), const(0, y), te.min_value(z)
    else:
        return const(0, x), te.min_value(y)


def min_combine(x, y):
    if len(x) == 3:
        eid = Select(x[2] < y[2], x[0], y[0])
        cid = Select(x[2] < y[2], x[1], y[1])
        val = Select(x[2] < y[2], x[2], y[2])
        return eid, cid, val
    else:
        idx = Select(x[1] < y[1], x[0], y[0])
        val = Select(x[1] < y[1], x[1], y[1])
        return idx, val


def min_identity(x, y, z=None):
    if z:
        return const(0, x), const(0, y), te.max_value(z)
    else:
        return const(0, x), te.max_value(y)


argmax = te.comm_reducer(max_combine, max_identity, name='argmax')
argmin = te.comm_reducer(min_combine, min_identity, name='argmin')

reduce_op_map = {
    'max': argmax,
    'min': argmin
}

binary_op_map = {
    'add': lambda x, y: x + y,
    'sub': lambda x, y: x - y,
    'mul': lambda x, y: x * y,
    'div': lambda x, y: x / y,
    'copy_lhs' : lambda x, y: x,
    'copy_rhs' : lambda x, y: y,
}

