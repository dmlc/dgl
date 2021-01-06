""" The compute function and schedules for GSDDMM kernels written in TVM. """
import tvm
from tvm import te
from tvm import topi
from tvm.topi.utils import prod, ravel_index, unravel_index
from tvm.tir import IntImm
from .utils import binary_op_map, TargetCode

__all__ = ['gsddmm']


def _sddmm_compute(out_shp, binary_op,
                   lhs, rhs, get_lhs, get_rhs):
    """Define the SDDMM' TVM compute function. """
    reduce_size = lhs.shape[-1] if binary_op == 'dot' else 1
    feat_len = prod(out_shp[1:])
    feat_len *= reduce_size
    if binary_op == 'dot':
        k = te.reduce_axis((0, reduce_size), name='k')
        def dot_edge_func(*args):
            eid = args[0]
            lval = lhs.__getitem__((get_lhs(eid),) + args[1: -1] + (k,))
            rval = rhs.__getitem__((get_rhs(eid),) + args[1: -1] + (k,))
            return te.sum(lval * rval, axis=k)
        out = te.compute(out_shp, dot_edge_func, name='out')
    else:
        def edge_func(*args):
            eid = args[0]
            lval = lhs.__getitem__((get_lhs(eid),) + args[1:])
            rval = rhs.__getitem__((get_rhs(eid),) + args[1:])
            return binary_op_map[binary_op](lval, rval)
        out = te.compute(out_shp, edge_func, name='out')
    return out


def _sddmm_cuda_general(sched, out):
    """Define the SDDMM's TVM general schedule. """
    out_len = prod(out.shape[1:])
    edge_axis = out.op.axis[0]
    feat_axis = sched[out].fuse(*out.op.axis[1:])
    #ntx = tvm.autotvm.task.space.get_pow2s(out_len)[-1]
    #ntx = 1024 if ntx > 1024 else ntx
    #nty = 1024 // ntx
    ntx = 32
    nty = 32
    feat_outer, feat_inner = sched[out].split(feat_axis, factor=ntx)
    edge_outer, edge_inner = sched[out].split(edge_axis, factor=nty)
    sched[out].bind(feat_inner, te.thread_axis('threadIdx.x'))
    sched[out].bind(feat_outer, te.thread_axis('blockIdx.y'))
    sched[out].bind(edge_inner, te.thread_axis('threadIdx.y'))
    sched[out].bind(edge_outer, te.thread_axis('blockIdx.x'))


def _sddmm_cuda_tree_reduce(sched, out):
    """Define the SDDMM's TVM tree-reduction schedule. """
    # TODO(zihao): handle other dimensions.
    edge_axis = out.op.axis[0]
    reduce_axis = out.op.reduce_axis[0]
    # sched[out].bind(reduce_axis, te.thread_axis('threadIdx.x'))
    # sched[out].bind(edge_axis, te.thread_axis('blockIdx.x'))
    _, red_inner = sched[out].split(reduce_axis, factor=32)
    edge_outer, edge_inner = sched[out].split(edge_axis, factor=32)
    sched[out].bind(red_inner, te.thread_axis('threadIdx.x'))
    sched[out].bind(edge_inner, te.thread_axis('threadIdx.y'))
    sched[out].bind(edge_outer, te.thread_axis('blockIdx.x'))


_sddmm_cuda_schedule = {
    'general': _sddmm_cuda_general,
    'tree': _sddmm_cuda_tree_reduce
}


def gsddmm(binary_op,
           ndim,
           indice_type, feat_type,
           lhs_target=TargetCode.SRC, rhs_target=TargetCode.DST,
           schedule_type='tree',
           target='cuda'):
    """
    Compile SDDMM kernel using TVM. 

    Parameters
    ----------
    binary_op : str
        Type of binary operatiin, could be ``add``, ``sub``, ``mul``,
        ``div`` or ``dot``.
    ndim : int
        Dimentionality.
    indice_type : str
        Type of graph indices, could be ``int32`` or ``int64``.
    feat_type : str
        Type of features, could be ``float16``/``float32``/``float64``
        or ``int32``/``int64``.
    lhs_target : TargetCode
        Indicates the left-hand-side tensor's target.
    rhs_target : TargetCode
        Indicates the right-hand-side tensor's target.
    schedule_type : str
        Specifies the schedule type, could be either ``tree`` or ``general``.
    target : str
        Indicates where kernels are run, i.e. CPU or GPU.

    Returns
    -------
    IRModule, representing compiled kernel. 
    """
    num_rows = te.var('num_rows', indice_type)
    num_cols = te.var('num_cols', indice_type)
    nnz = te.var('nnz', indice_type)

    # placeholder for sparse matrix
    adj_row_indices = te.placeholder((nnz,), indice_type, 'adj_row_indices')
    adj_col_indices = te.placeholder((nnz,), indice_type, 'adj_col_indices')

    # placeholder for dense features
    def create_placeholder(target, feat_shp, name):
        if target == TargetCode.SRC:
            return te.placeholder((num_rows,) + feat_shp, feat_type, name)
        elif target == TargetCode.EDGE:
            return te.placeholder((nnz,) + feat_shp, feat_type, name)
        elif target == TargetCode.DST:
            return te.placeholder((num_cols,) + feat_shp, feat_type, name)
        else:
            raise DGLError('Unknown target')

    out_feat_shp = [te.var('d_o{}'.format(i), indice_type) for i in range(ndim)]
    lhs_feat_shp = [te.var('d_l{}'.format(i), indice_type) for i in range(ndim)]
    rhs_feat_shp = [te.var('d_r{}'.format(i), indice_type) for i in range(ndim)]
    lhs = create_placeholder(lhs_target, tuple(lhs_feat_shp), 'lhs')
    rhs = create_placeholder(rhs_target, tuple(rhs_feat_shp), 'rhs')

    # idx wrapper for corresponding target
    target_getter = {
        TargetCode.SRC: lambda eid: adj_row_indices[eid],
        TargetCode.EDGE: lambda eid: eid,
        TargetCode.DST: lambda eid: adj_col_indices[eid]
    }

    # compute
    out = _sddmm_compute([nnz] + out_feat_shp,
                         binary_op, lhs, rhs,
                         target_getter[lhs_target], target_getter[rhs_target])

    # schedule
    sched = te.create_schedule(out.op)

    if target == 'cuda':
        _sddmm_cuda_schedule[schedule_type](sched, out)
    elif target == 'llvm':
        raise NotImplementedError('CPU kernel not implemented yet.')

    # prepare input
    f_input = out_feat_shp
    f_input.append(adj_row_indices)
    f_input.append(adj_col_indices)
    f_name = '_'.join(str(x) for x in [
        'sddmm', binary_op, ndim,
        indice_type, feat_type,
        lhs_target, rhs_target, schedule_type, target])
    f_input += [lhs, rhs, out]

    # bind autobroadcast buffer
    lhs_buffer = tvm.tir.decl_buffer(lhs.shape, lhs.dtype, name='lhs_buf',
                                     buffer_type='auto_broadcast')
    rhs_buffer = tvm.tir.decl_buffer(rhs.shape, rhs.dtype, name='rhs_buf',
                                     buffer_type='auto_broadcast')
    binds = {lhs:lhs_buffer, rhs:rhs_buffer}
    return tvm.lower(sched, f_input, name=f_name, binds=binds)
