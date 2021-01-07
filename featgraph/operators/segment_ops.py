""" The compute function and schedules for segment operators written in TVM. """
import tvm
from tvm import te
from tvm import topi
from tvm.tir import IntImm
from .utils import reduce_op_map, TargetCode

__all__ = ['segment_reduce']


def _segment_reduce(out_shp, reduce_op,
                    offsets, in_feat):
    require_arg = reduce_op in ['min', 'max']
    n = offsets.shape[0] - 1
    def reduce_func(i, j):
        start = offsets[i]
        end = offsets[i + 1]
        # The following line avoids the possible illegal memory access error
        # Related issue: https://github.com/apache/tvm/issues/6596
        length = te.if_then_else(i < n, end - start, 0)
        k = te.reduce_axis((0, length), name='k')
        row = k + start
        val = []
        if require_arg:
            val.append(j)
        val.append(in_feat[row, j])
        return reduce_op_map[reduce_op](val, axis=k)
    
    ret = te.compute(out_shp, reduce_func, name='out')
    if reduce_op == 'sum':
        ret = (ret,)
    return ret


def _segment_cuda_general(sched, out):
    #out_len = prod(out.shape[1:])
    segment_axis = out.op.axis[0]
    feat_axis = sched[out].fuse(*out.op.axis[1:])
    sched[out].bind(feat_axis, te.thread_axis('threadIdx.x'))
    sched[out].bind(segment_axis, te.thread_axis('blockIdx.x')) 


def _segment_cuda_tree_reduce(sched, out):
    # TODO(zihao): first transpose then schedule
    """
    reduce_axis = out.op.reduce_axis[0]
    segment_axis = out.op.axis[0]
    ret_outer, red_inner = sched[out].split(reduce_axis, factor=32)
    sched[out].bind(red_inner, te.thread_axis('threadIdx.x'))
    """
    pass


_segment_reduce_cuda_schedule = {
    'general': _segment_cuda_general,
    'tree': _segment_cuda_tree_reduce
}


def segment_reduce(reduce_op,
                   indice_type, feat_type,
                   schedule_type='general',
                   target='cuda'):
    """
    Compile Segment Reduce kernels using TVM.
    """
    m = te.var('m', indice_type)
    n = te.var('n', indice_type)
    offsets = te.placeholder((n + 1,), indice_type, 'offsets')
    d = te.var('d', indice_type)
    in_feat = te.placeholder([m, d], feat_type, 'in_feat')
    ret = _segment_reduce([n, d], reduce_op, offsets, in_feat)
    out = ret[-1]
    sched = te.create_schedule(out.op)
    if target == 'cuda':
        _segment_reduce_cuda_schedule[schedule_type](sched, out)
    else:
        raise NotImplementedError("CPU schedule not implemented yet.")
    f_name = '_'.join(str(x) for x in [
        'segment_reduce', reduce_op,
        indice_type, feat_type,
        schedule_type, target
    ])
    f_input = [offsets, in_feat]
    f_input += ret

    return tvm.lower(sched, f_input, name=f_name)
