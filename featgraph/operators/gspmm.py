""" The compute function and schedules for GSpMM kernels written in TVM. """
import tvm
from tvm import te
from tvm import topi
from tvm.tir import IntImm
from .utils import binary_op_map, reduce_op_map, TargetCode


def _spmm(out_shp, binary_op, reduce_op,
          indptr, indices,
          ufeat, efeat):
    use_u = binary_op != 'copy_rhs'
    use_e = binary_op != 'copy_lhs'
    require_arg = reduce_op in ['min', 'max']
    n = indptr.shape[0] - 1
    def node_func(*args):
        row = args[0]
        start = indptr[row]
        end = indptr[row + 1]
        # The following line avoids the possible illegal memory access error
        # Related issue: https://github.com/apache/tvm/issues/6596
        length = te.if_then_else(row < n, end - start, 0)
        k = te.reduce_axis((0, length), name='k')
        uid = indices[eid]
        eid = k + start
        msg = []
        if use_e:
            e_val = efeat.__getitem__((eid,) + args[1:])
            msg.append(eid)
        if use_u:
            u_val = ufeat.__getitem__((uid,) + args[1:])
            msg.append(uid)
        if binary_op == 'copy_lhs':
            msg.append(u_val)
        elif binary_op == 'copy_rhs':
            msg.append(e_val)
        else:
            msg.append(binary_op_map[binary_op](u_val, e_val))
        return reduce_op_map[reduce_op](msg, axis=k)
        
    ret = te.compute(out_shp, node_func, name='out')
    if reduce_op == 'sum':
        ret = (ret,)
    return ret
