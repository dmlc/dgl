from tvm.tir import Select, const
from tvm import te

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
    'add': lambda x, y: x+y,
    'sub': lambda x, y: x-y,
    'mul': lambda x, y: x*y,
    'div': lambda x, y: x/y,
    'copy_lhs' : lambda x, y: x,
    'copy_rhs' : lambda x, y: y,
}
