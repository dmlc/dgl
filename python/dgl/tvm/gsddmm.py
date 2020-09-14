import tvm
from tvm import te
from tvm import topi
from tvm.tir import IntImm

from ..base import DGLError
from .util import binary_op_map
from ..function import TargetCode

def _sddmm_compute(out_shp, binary_op, lhs, rhs,
                   lhs_idx, rhs_idx, num_feat_partitions=1,
                   lhs_pack=False, rhs_pack=False):
    reduce_size = lhs.shape[-1] if binary_op == 'dot' else 1
    feat_len = topi.util.get_const_int(topi.util.prod(out_shp[1:]))
    feat_len *= reduce_size
    # assume feat_len is a multiply of num_feat_partitions
    feat_len_per_partition = feat_len // num_feat_partitions
    reshapes = []
    def reshape(feat_outer, n, feat_inner, tensor):
        fid = feat_outer * feat_len_per_partition + feat_inner
        return tensor.__getitem__((n,) + tuple(topi.util.unravel_index(fid, tensor.shape[1:])))
    if lhs_pack:
        reshaped_lhs = te.compute((num_feat_partitions, lhs.shape[0], feat_len_per_partition),
                                  lambda feat_outer, idx, feat_inner:
                                  reshape(feat_outer, idx, feat_inner, lhs),
                                  name='reshaped_lhs')
        reshapes.append(reshaped_lhs)
    if rhs_pack:
        reshaped_rhs = te.compute((num_feat_partitions, rhs.shape[0], feat_len_per_partition),
                                  lambda feat_outer, idx, feat_inner:
                                  reshape(feat_outer, idx, feat_inner, rhs),
                                  name='reshaped_rhs')
        reshapes.append(reshaped_rhs)
    if binary_op == 'dot':
        k = te.reduce_axis((0, reduce_size), name='k')
        def dot_edge_func(*args):
            eid = args[0]
            fid = topi.util.ravel_index(args[1:], out_shp[1:])
            fid *= reduce_size
            if lhs_pack:
                lval = reshaped_lhs[(fid + k) // feat_len_per_partition, \
                            lhs_idx(eid), (fid + k) % feat_len_per_partition]
            else:
                lval = lhs.__getitem__((lhs_idx(eid),) + args[1:-1] +(k,))
            if rhs_pack:
                rval = reshaped_rhs[(fid + k) // feat_len_per_partition, \
                            rhs_idx(eid), (fid + k) % feat_len_per_partition]
            else:
                rval = rhs.__getitem__((rhs_idx(eid),) + args[1:-1] +(k,))
            return te.sum(lval * rval, axis=k)
        out = te.compute(out_shp, dot_edge_func, name='out')
    else:
        def edge_func(*args):
            eid = args[0]
            fid = topi.util.ravel_index(args[1:], out_shp[1:])
            if lhs_pack:
                lval = reshaped_lhs[fid // feat_len_per_partition, \
                             lhs_idx(eid), fid % feat_len_per_partition]
            else:
                lval = lhs.__getitem__((lhs_idx(eid),) + args[1:])
            if rhs_pack:
                rval = reshaped_rhs[fid // feat_len_per_partition, \
                             rhs_idx(eid), fid % feat_len_per_partition]
            else:
                rval = rhs.__getitem__((rhs_idx(eid),) + args[1:])
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

def _sddmm_cpu_general(sched, out):
    edge_axis = out.op.axis[0]
    sched[out].parallel(edge_axis)

def _sddmm_cpu_feat_partition(sched, out, op, reshapes, reduce_size, num_feat_partitions):
    edge_axis = out.op.axis[0]
    feat_axis = sched[out].fuse(*out.op.axis[1:])
    feat_len = topi.util.get_const_int(topi.util.prod(out.shape[1:])) * reduce_size
    feat_len_per_partition = feat_len // num_feat_partitions
    outermost = None
    if op != 'dot':
        feat_outer, feat_inner = sched[out].split(feat_axis, factor=feat_len_per_partition)
        sched[out].reorder(feat_outer, edge_axis, feat_inner)
        outermost = feat_outer
    else:
        reduce_axis = out.op.reduce_axis[0]
        if reduce_size == feat_len_per_partition:
            sched[out].reorder(feat_axis, edge_axis, reduce_axis)
            outermost = feat_len
        elif reduce_size < feat_len_per_partition:
            feat_outer, feat_inner = \
                sched[out].split(feat_axis, factor=feat_len_per_partition // reduce_size)
            sched[out].reorder(feat_outer, edge_axis, feat_inner, reduce_axis)
            outermost = feat_outer
        else:
            red_outer, red_inner = sched[out].split(reduce_axis, factor=feat_len_per_partition)
            sched[out].reorder(red_outer, feat_axis, edge_axis, red_inner)
            outermost = red_outer
    for reshape in reshapes:
        sched[reshape].compute_at(sched[out], outermost)
        sched[reshape].parallel(reshape.op.axis[1])
    sched[out].parallel(edge_axis)

def sddmm(binary_op, nnz, num_rows, num_cols,
          lhs_shp, rhs_shp, out_shp, indice_type, feat_type,
          lhs_target=0, rhs_target=2, is_sorted=2, use_idx=False,
          target='llvm', num_feat_partitions=1):
    if binary_op in ['copy_lhs', 'copy_rhs']:
        raise DGLError('SDDMM {} op should not use tvm kernel'.format(binary_op))
    if '32' in indice_type:
        indice_type = 'int32'
    elif '64' in indice_type:
        indice_type = 'int64'
    else:
        raise DGLError('Unrecognized number of bits')
    if '16' in feat_type:
        feat_type = 'float16'
    elif '32' in feat_type:
        feat_type = 'float32'
    elif '64' in feat_type:
        feat_type = 'float64'
    else:
        raise DGLError('Unrecognized number of bits')
    # check if use generic shape
    generic_shape = nnz == 0 and num_rows == 0 and num_cols == 0
    if generic_shape:
        num_rows = te.var('num_rows', indice_type)
        num_cols = te.var('num_cols', indice_type)
        nnz = te.var('nnz', indice_type)
    else:
        # convert python int into tir nodes so that type of iterator in generated code is correct
        num_rows = IntImm(indice_type, num_rows)
        num_cols = IntImm(indice_type, num_cols)
        nnz = IntImm(indice_type, nnz)
    # check if use bcast
    use_bcast = lhs_shp != rhs_shp
    # check should be done in infer_out_shape
    if binary_op == 'dot':
        reduce_size = lhs_shp[-1]
    else:
        reduce_size = 1
    # placeholder for sparse matrix
    adj_row_indices = te.placeholder((nnz,), indice_type, 'adj_row_indices')
    adj_col_indices = te.placeholder((nnz,), indice_type, 'adj_col_indices')
    edge_mapping = te.placeholder((nnz,), indice_type, 'edge_mapping')
    # placeholder for dense features
    def switch_target(target, feat_shp, name):
        if target == TargetCode.SRC:
            return te.placeholder((num_rows,) + feat_shp, feat_type, name)
        elif target == TargetCode.EDGE:
            return te.placeholder((nnz,) + feat_shp, feat_type, name)
        elif target == TargetCode.DST:
            return te.placeholder((num_cols,) + feat_shp, feat_type, name)
        else:
            raise DGLError('Unknown target')
    lhs = switch_target(lhs_target, tuple([IntImm(indice_type, s) for s in lhs_shp]), 'lhs')
    rhs = switch_target(rhs_target, tuple([IntImm(indice_type, s) for s in rhs_shp]), 'rhs')
    # idx wrapper for corresponding target
    def idx_target(target):
        def idx(eid):
            if target == TargetCode.SRC:
                return adj_row_indices[eid]
            elif target == TargetCode.EDGE:
                return edge_mapping[eid] if use_idx else eid
            elif target == TargetCode.DST:
                return adj_col_indices[eid]
            else:
                raise DGLError('Unknown target')
        return idx
    # if feature partition, decide whether to apply array packing
    # sorted = 0 means row-sorted, 1 means col-sorted, 2 means not sorted
    # if edge_mapping is present, do not use array packing
    lhs_pack = (num_feat_partitions > 1 and not use_bcast) and \
               ((lhs_target == TargetCode.SRC and is_sorted == 0) or \
                (lhs_target == TargetCode.DST and is_sorted == 1) or \
                (lhs_target == TargetCode.EDGE and not use_idx))
    rhs_pack = (num_feat_partitions > 1 and not use_bcast) and \
               ((rhs_target == TargetCode.SRC and is_sorted == 0) or \
                (rhs_target == TargetCode.DST and is_sorted == 1) or \
                (rhs_target == TargetCode.EDGE and not use_idx))
    # compute
    out, reshapes = _sddmm_compute((nnz,) + tuple([IntImm(indice_type, s) for s in out_shp]),
                                   binary_op, lhs, rhs,
                                   idx_target(lhs_target), idx_target(rhs_target),
                                   num_feat_partitions, lhs_pack, rhs_pack)
    # schedule
    sched = te.create_schedule(out.op)
    if target == 'cuda':
        # cuda schedule
        if binary_op == 'dot' and reduce_size >= 32:
            # if dot product, use tree reduction
            _sddmm_cuda_tree_reduce(sched, out)
        else:
            _sddmm_cuda_general(sched, out)
    elif target == 'llvm':
        if num_feat_partitions == 1:
            _sddmm_cpu_general(sched, out)
        else:
            _sddmm_cpu_feat_partition(sched, out, binary_op, reshapes,
                                      reduce_size, num_feat_partitions)
    # prepare input
    f_input = []
    if lhs_target == TargetCode.SRC or rhs_target == TargetCode.SRC:
        f_input.append(adj_row_indices)
    if lhs_target == TargetCode.DST or rhs_target == TargetCode.DST:
        f_input.append(adj_col_indices)
    if use_idx:
        f_input.append(edge_mapping)
    f_name = '_'.join(str(x) for x in [
        'sddmm', binary_op,
        lhs_target, rhs_target,
        indice_type, feat_type])
    f_input += [lhs, rhs, out]
    # bind autobroadcast buffer
    lhs_buffer = tvm.tir.decl_buffer(lhs.shape, lhs.dtype, name='lhs_buf',
                                     buffer_type='auto_broadcast')
    rhs_buffer = tvm.tir.decl_buffer(rhs.shape, rhs.dtype, name='rhs_buf',
                                     buffer_type='auto_broadcast')
    binds = {}
    if use_bcast:
        binds = {lhs:lhs_buffer, rhs:rhs_buffer}
    # print(tvm.lower(sched, f_input, binds=binds))
    return tvm.build(sched, f_input, target=target, name=f_name, binds=binds)
