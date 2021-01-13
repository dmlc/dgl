""" The compute function and schedules for segment operators written in TVM. """
import tvm
from tvm import te, tir
from tvm import topi
from .utils import reduce_op_map, TargetCode

__all__ = ['segment_reduce', 'segment_gemm']


def _segment_reduce(out_shp, reduce_op,
                    offsets, in_feat):
    require_arg = reduce_op in ['min', 'max']
    m = in_feat.shape[0]
    n = offsets.shape[0] - 1
    d = in_feat.shape[1]
    in_trans = te.compute((d, m), lambda i, j: in_feat[j, i], name='in_trans')

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
        val.append(in_trans[j, row])
        return reduce_op_map[reduce_op](val, axis=k)
    
    ret = te.compute(out_shp, reduce_func, name='out')
    if reduce_op == 'sum':
        ret = (ret,)
    return in_trans, ret


def _segment_gemm(n_arr, m_arr, p_arr,
                  a_off_arr, b_off_arr, c_off_arr,
                  A, B, C):
    num_segs = n_arr.shape[0]
    ib = tir.ir_builder.create()

    n_arr_ptr = ib.buffer_ptr(n_arr)
    m_arr_ptr = ib.buffer_ptr(m_arr)
    p_arr_ptr = ib.buffer_ptr(p_arr)
    a_off_arr_ptr = ib.buffer_ptr(a_off_arr)
    b_off_arr_ptr = ib.buffer_ptr(b_off_arr)
    c_off_arr_ptr = ib.buffer_ptr(c_off_arr)
    A_ptr = ib.buffer_ptr(A)
    B_ptr = ib.buffer_ptr(B)
    C_ptr = ib.buffer_ptr(C)

    with ib.for_range(0, num_segs, name='seg_idx') as seg_idx:
        n = n_arr_ptr[seg_idx]
        m = m_arr_ptr[seg_idx]
        p = p_arr_ptr[seg_idx]
        a_off = a_off_arr_ptr[seg_idx]
        b_off = b_off_arr_ptr[seg_idx]
        c_off = c_off_arr_ptr[seg_idx]
        with ib.for_range(0, n, name='i') as i:
            with ib.for_range(0, p, name='j') as j:
                C_ptr[c_off + (i * p + j)] = tir.FloatImm(A.dtype, 0.)
                with ib.for_range(0, m, name='k') as k:
                    C_ptr[c_off + (i * p + j)] += A_ptr[a_off + (i * m + k)] * B_ptr[b_off + (k * p + j)]
    
    return ib.get()


def _segment_cuda_general(sched, in_trans, out):
    sched[in_trans].compute_inline()
    segment_axis = out.op.axis[0]
    feat_axis = out.op.axis[1]
    sched[out].bind(feat_axis, te.thread_axis('threadIdx.x'))
    sched[out].bind(segment_axis, te.thread_axis('blockIdx.x')) 


def _segment_cuda_tree_reduce(sched, in_trans, out):
    s_in = sched[in_trans]
    segment_axis = in_trans.op.axis[0]
    feat_axis = in_trans.op.axis[1]
    s_in.bind(segment_axis, te.thread_axis('blockIdx.x'))
    s_in.bind(feat_axis, te.thread_axis('threadIdx.x'))

    s_out = sched[out]
    reduce_axis = out.op.reduce_axis[0]
    segment_axis = out.op.axis[0]
    feat_axis = out.op.axis[1]
    feat_outer, feat_inner = s_out.split(feat_axis, factor=32)
    _, red_inner = s_out.split(reduce_axis, factor=32)
    s_out.bind(red_inner, te.thread_axis('threadIdx.x'))
    s_out.bind(feat_inner, te.thread_axis('threadIdx.y'))
    s_out.bind(segment_axis, te.thread_axis('blockIdx.x'))
    s_out.bind(feat_outer, te.thread_axis('blockIdx.y'))


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
    in_trans, ret = _segment_reduce([n, d], reduce_op, offsets, in_feat)
    out = ret[-1]
    sched = te.create_schedule(out.op)
    if target == 'cuda':
        _segment_reduce_cuda_schedule[schedule_type](sched, in_trans, out)
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


def segment_gemm(indice_type, feat_type):
    num_segs = te.var('num_segs') 
    n_arr = te.placeholder((num_segs,), indice_type, 'n_arr')
    m_arr = te.placeholder((num_segs,), indice_type, 'm_arr')
    p_arr = te.placeholder((num_segs,), indice_type, 'p_arr')
    a_off_arr = te.placeholder((num_segs + 1,), indice_type, 'a_off_arr')
    b_off_arr = te.placeholder((num_segs + 1,), indice_type, 'b_off_arr')
    c_off_arr = te.placeholder((num_segs + 1,), indice_type, 'c_off_arr')
    A_numel = te.var('A_numel')
    A = te.placeholder((A_numel,), feat_type, 'A')
    B_numel = te.var('B_numel')
    B = te.placeholder((B_numel,), feat_type, 'B')
    C_numel = te.var('C_numel')
    #C = _segment_gemm(C_numel, n_arr, m_arr, p_arr, a_off_arr, b_off_arr, c_off_arr, A, B)
    C = te.extern([C_numel],
                  [n_arr, m_arr, p_arr, a_off_arr, b_off_arr, c_off_arr, A, B],
                  lambda ins, outs: _segment_gemm(*ins, *outs),
                  dtype=feat_type,
                  name='C')
    sched = te.create_schedule(C.op)
    f_name = '_'.join(str(x) for x in [
        'segment_gemm', indice_type, feat_type
    ])
    f_input = [n_arr, m_arr, p_arr, a_off_arr, b_off_arr, c_off_arr, A, B, C]
    return tvm.lower(sched, f_input, name=f_name)
