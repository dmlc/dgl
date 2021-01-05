""" The compute function and schedules for SDDMM kernels written in TVM. """
import tvm
from tvm import te
from tvm import topi
from tvm.tir import IntImm
from .util import binary_op_map

__all__ = ['gsddmm']


def _sddmm_compute(out_shp, binary_op,
                   lhs, rhs,
                   lhs_idx, rhs_idx):
    reduce_size = lhs.shape[-1] if binary_op == 'dot' else 1
    feat_len = topi.util.get_const_int(topi.util.prod(out_shp[1:]))
    feat_len *= reduce_size
    # assume feat_len is a multiply of num_feat_partitions
    feat_len_per_partition = feat_len // num_feat_partitions
    if binary_op == 'dot':
        k = te.reduce_axis((0, reduce_size), name='k')
        def dot_edge_func(*args):
            eid = args[0]
            fid = topi.util.ravel_index(args[1:], out_shp[1:])
            fid *= reduce_size
            lval = lhs((lhs_idx(eid),) + args[1: -1] + (k,))
            rval = rhs((rhs_idx(eid),) + args[1: -1] + (k,))
            return te.sum(lval * rval, axis=k)
        out = te.compute(out_shp, dot_edge_func, name='out')
    else:
        def edge_func(*args):
            eid = args[0]
            fid = topi.util.ravel_index(args[1:], out_shp[1:])
            lval = lhs((lhs_idx(eid),) + args[1:])
            rval = rhs((rhs_idx(eid),) + args[1:])
            return binary_op_map[binary_op](lval, rval)
        out = te.compute(out_shp, edge_func, name='out')
    return out, reshapes


def _sddmm_cuda_general(sched, out):
    out_len = topi.util.get_const_int(topi.util.prod(out.shape[1:]))
    edge_axis = out.op.axis[0]
    feat_axis = sched[out].fuse(*out.op.axis[1:])
    ntx = tvm.autotvm.task.space.get_pow2s(out_len)[-1]
    ntx = 1024 if ntx > 1024 else ntx
    nty = 1024 // ntx
    feat_outer, feat_inner = sched[out].split(feat_axis, factor=ntx)
    edge_outer, edge_inner = sched[out].split(edge_axis, factor=nty)
    sched[out].bind(feat_inner, te.thread_axis('threadIdx.x'))
    sched[out].bind(feat_outer, te.thread_axis('blockIdx.y'))
    sched[out].bind(edge_inner, te.thread_axis('threadIdx.y'))
    sched[out].bind(edge_outer, te.thread_axis('blockIdx.x'))


def _sddmm_cuda_tree_reduce(sched, out):
    edge_axis = out.op.axis[0]
    reduce_axis = out.op.reduce_axis[0]
    # sched[out].bind(reduce_axis, te.thread_axis('threadIdx.x'))
    # sched[out].bind(edge_axis, te.thread_axis('blockIdx.x'))
    _, red_inner = sched[out].split(reduce_axis, factor=32)
    edge_outer, edge_inner = sched[out].split(edge_axis, factor=32)
    sched[out].bind(red_inner, te.thread_axis('threadIdx.x'))
    sched[out].bind(edge_inner, te.thread_axis('threadIdx.y'))
    sched[out].bind(edge_outer, te.thread_axis('blockIdx.x'))


def gsddmm(binary_op, ndim,
           indice_type, feat_type,
           lhs_target=TargetCode.SRC, rhs_target=TargetCode.DST,
           use_edge_idx=False,
           schedule_type="tree",
           target='cuda'):
    """
    Compile SDDMM kernel using TVM. 

    Parameters
    ----------
    binary_op : str
        Type of binary operatiin, could be ``add``, ``sub``, ``mul``,
        ``div`` or ``dot``.
    ndim : int
        The number of feature dimensions.
    indice_type : str
        Type of graph indices, could be ``int32`` or ``int64``.
    feat_type : str
        Type of features, could be ``float16``/``float32``/``float64``
        or ``int32``/``int64``.
    lhs_target : TargetCode
        Indicates the left-hand-side tensor's target.
    rhs_target : TargetCode
        Indicates the right-hand-side tensor's target.
    use_edge_idx : bool
        Indicates whether to use edge index or not.
    schedule_type : str
        Specifies the schedule type.
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
    edge_indices = te.placeholder((nnz,), indice_type, 'edge_indices')

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

    def shfl_edge_feat(feat, name):
        def _shfl(*args):
            eid = args[0]
            return feat((edge_indices[eid],) + args[1:])
        return te.compute(feat.shape, _shfl, name=name)

    lhs_feat_shp = [te.var('lhs_dim_{}'.format(i), indice_type) for i in range(ndim)]
    rhs_feat_shp = [te.var('rhs_dim_{}'.format(i), indice_type) for i in range(ndim)]
    lhs = create_placeholder(lhs_target, tuple(lhs_feat_shp), 'lhs')
    rhs = create_placeholder(rhs_target, tuple(rhs_feat_shp), 'rhs')
    if use_idx:
        if lhs_target == TargetCode.EDGE:
            lhs_shfl = shfl_edge_feat(lhs, 'lhs_shfl') 
        else:
            lhs_shfl = None
        if rhs_target == TargetCode.EDGE:
            rhs_shfl = shfl_edge_feat(rhs, 'rhs_shfl')
        else:
            rhs_shlf = None

    # idx wrapper for corresponding target
    idx_target = {
        TargetCode.SRC: lambda eid: adj_row_indices[eid],
        TargetCode.EDGE: lambda eid: eid,
        TargetCode.DST: lambda eid: adj_col_indices[eid]
    }

    # compute
    out = _sddmm_compute((nnz,) + tuple([IntImm(indice_type, s) for s in out_shp]),
                         binary_op, lhs_shfl if lhs_shfl else lhs, rhs_shfl if rhs_shfl else rhs,
                         idx_target[lhs_target], idx_target[rhs_target])

    # schedule
    sched = te.create_schedule(out.op)
    if lhs_shfl:
        sched[lhs_shfl].compute_inilne()
    if rhs_shfl:
        sched[rhs_shfl].compute_inline()
    if target == 'cuda':
        # cuda schedule
        if schedule_type == 'tree':
            _sddmm_cuda_tree_reduce(sched, out)
        else:
            _sddmm_cuda_general(sched, out)
    elif target == 'llvm':
        raise NotImplementedError('CPU kernel not implemented yet.')

    # prepare input
    f_input = lhs_shp + rhs_shp
    f_input.append(adj_row_indices)
    f_input.append(adj_col_indices)
    if use_idx:
        f_input.append(edge_indices)
    f_name = '_'.join(str(x) for x in [
        'sddmm', binary_op, ndim,
        indice_type, feat_type
        lhs_target, rhs_target,
        use_edge_idx, schedule_type])
    f_input += [lhs, rhs, out]
    # bind autobroadcast buffer
    lhs_buffer = tvm.tir.decl_buffer(lhs.shape, lhs.dtype, name='lhs_buf',
                                     buffer_type='auto_broadcast')
    rhs_buffer = tvm.tir.decl_buffer(rhs.shape, rhs.dtype, name='rhs_buf',
                                     buffer_type='auto_broadcast')
    binds = {}
    if use_bcast:
        binds = {lhs:lhs_buffer, rhs:rhs_buffer}
    return tvm.lower(sched, f_input, name=f_name, binds=binds)
