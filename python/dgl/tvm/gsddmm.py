import tvm
from tvm import te
from tvm import topi
from .util import binary_op_map
from ..function import TargetCode

def _sddmm_compute(out_shp, binary_op, lhs, rhs, 
                   lhs_idx, rhs_idx, num_feat_partitions=1,
                   lhs_pack=False, rhs_pack=False):
    reduce_size = lhs.shape[-1] if binary_op == 'dot' else 1
    feat_len = 1
    for d in out_shp[1:]:
        feat_len *= d
    feat_len *= reduce_size
    # assume feat_len is a multiply of num_feat_partitions
    feat_len_per_partition = feat_len // num_feat_partitions
    reshapes = []
    def reshape(fo, n, fi, t, idx):
        ff = fo * feat_len_per_partition + fi
        return t.__getitem__((idx(n, True),) + tuple(topi.util.unravel_index(ff, t.shape[1:])))
    if lhs_pack:
        reshaped_lhs = te.compute((num_feat_partitions, lhs.shape[0], feat_len_per_partition), \
                                  lambda fo, idx, fi: reshape(fo, idx, fi, lhs, lhs_idx), name='reshaped_lhs')
        reshapes.append(reshaped_lhs)
    if rhs_pack:
        reshaped_rhs = te.compute((num_feat_partitions, rhs.shape[0], feat_len_per_partition), \
                                  lambda fo, idx, fi: reshape(fo, idx, fi, rhs, rhs_idx), name='reshaped_rhs')
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
                lval = lhs.__getitem__((lhs_idx(args[0]),) + args[1:-1] +(k,))
            if rhs_pack:
                rval = reshaped_rhs[(fid + k) // feat_len_per_partition, \
                            rhs_idx(eid), (fid + k) % feat_len_per_partition]
            else:
                rval = rhs.__getitem__((rhs_idx(args[0]),) + args[1:-1] +(k,))
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
                lval = lhs.__getitem__((lhs_idx(args[0]),) + args[1:])
            if rhs_pack:
                rval = reshaped_rhs[fid // feat_len_per_partition, \
                             rhs_idx(eid), fid % feat_len_per_partition]
            else:
                rval = rhs.__getitem__((rhs_idx(args[0]),) + args[1:])
            return binary_op_map[binary_op](lval, rval)
        out = te.compute(out_shp, edge_func, name='out')
    return out, reshapes

def _sddmm_cuda_general(s, out):
    out_len = topi.util.get_const_int(topi.util.prod(out.shape[1:]))
    edge_axis = out.op.axis[0]
    feat_axis = s[out].fuse(*out.op.axis[1:])
    ntx = tvm.autotvm.task.space.get_pow2s(out_len)[-1]
    ntx = 1024 if ntx > 1024 else ntx
    nty = 1024 // ntx
    fo, fi = s[out].split(feat_axis, factor=ntx)
    eo, ei = s[out].split(edge_axis, factor=nty)
    s[out].bind(fi, te.thread_axis('threadIdx.x'))
    s[out].bind(fo, te.thread_axis('blockIdx.y'))
    s[out].bind(ei, te.thread_axis('threadIdx.y'))
    s[out].bind(eo, te.thread_axis('blockIdx.x'))

def _sddmm_cuda_tree_reduce(s, out):
    edge_axis = out.op.axis[0]
    reduce_axis = out.op.reduce_axis[0]
    # eo, ei = s[out].split(edge_axis, factor = (1024 // reduce_size))
    # s[out].bind(reduce_axis, te.thread_axis('threadIdx.x'))
    ro, ri = s[out].split(reduce_axis, factor = 32)
    eo, ei = s[out].split(edge_axis, factor = 32)
    s[out].bind(ri, te.thread_axis('threadIdx.x'))
    s[out].bind(ei, te.thread_axis('threadIdx.y'))
    s[out].bind(eo, te.thread_axis('blockIdx.x'))

def _sddmm_cpu_general(s, out):
    edge_axis = out.op.axis[0]
    s[out].parallel(edge_axis)
    
def _sddmm_cpu_feat_partition(s, out, op, reshapes, reduce_size, num_feat_partitions):
    edge_axis = out.op.axis[0]
    feat_axis = s[out].fuse(*out.op.axis[1:])
    feat_len = topi.util.get_const_int(topi.util.prod(out.shape[1:])) * reduce_size
    feat_len_per_partition = feat_len // num_feat_partitions
    outermost = None
    if op != 'dot':
        fo, fi = s[out].split(feat_axis, factor=feat_len_per_partition)
        s[out].reorder(fo, edge_axis, fi)
        outermost = fo
    else:
        reduce_axis = out.op.reduce_axis[0]
        if reduce_size == feat_len_per_partition:
            s[out].reorder(feat_axis, edge_axis, reduce_axis)
            outermost = feat_len
        elif reduce_size < feat_len_per_partition:
            fo, fi = s[out].split(feat_axis, factor=feat_len_per_partition // reduce_size)
            s[out].reorder(fo, edge_axis, fi, reduce_axis)
            outermost = fo
        else:
            ro, ri = s[out].split(reduce_axis, factor=feat_len_per_partition)
            s[out].reorder(ro, feat_axis, edge_axis, ri)
            outermost = ro
    for t in reshapes:
        s[t].compute_at(s[out], outermost)
        s[t].parallel(t.op.axis[1])
    s[out].parallel(edge_axis)

def sddmm(binary_op, nnz, num_rows, num_cols, 
          lhs_shp, rhs_shp, out_shp, indice_type, feat_type,
          lhs_target=0, rhs_target=2, is_sorted=2, use_idx=False,
          target='llvm', num_feat_partitions=1):
    if binary_op in ['copy_lhs', 'copy_rhs']:
        raise NotImplementedError
    if '32' in indice_type:
        indice_type = 'int32'
    elif '64' in indice_type:
        indice_type = 'int64'
    else:
        raise NotImplementedError
    if '16' in feat_type:
        feat_type = 'float16'
    elif '32' in feat_type:
        feat_type = 'float32'
    elif '64' in feat_type:
        feat_type = 'float64'
    else:
        raise NotImplementedError
    # check if use generic shape
    generic_shape = nnz == 0 and num_rows == 0 and num_cols == 0
    if generic_shape:
        num_rows = te.var('num_rows', indice_type)
        num_cols = te.var('num_cols', indice_type)
        nnz = te.var('nnz', indice_type)
    else:
        # convert python int into tir nodes so that type of iterator in generated code is correct
        num_rows = tvm.tir.IntImm(indice_type, num_rows)
        num_cols = tvm.tir.IntImm(indice_type, num_cols)
        nnz = tvm.tir.IntImm(indice_type, nnz)
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
    def switch_target(t, s, name):
        if t == TargetCode.SRC:
            return te.placeholder((num_rows,) + s, feat_type, name)
        elif t == TargetCode.EDGE:
            return te.placeholder((nnz,) + s, feat_type, name)
        elif t == TargetCode.DST:
            return te.placeholder((num_cols,) + s, feat_type, name)
    lhs = switch_target(lhs_target, lhs_shp, 'lhs')
    rhs = switch_target(rhs_target, rhs_shp, 'rhs')
    # idx wrapper for corresponding target
    # if use_idx and use packing, edge_mapping can be done in packing phase to save memory
    def idx_target(t):
        def foo(eid, pack=False):
            if pack:
                return edge_mapping[eid] if use_idx and t == 1 else eid
            if t == TargetCode.SRC:
                return adj_row_indices[eid]
            elif t == TargetCode.EDGE:
                return eid
            elif t == TargetCode.DST:
                return adj_col_indices[eid]
        return foo
    # if feature partition, decide whether to apply array packing
    # sorted = 0 means row-sorted, 1 means col-sorted, 2 means not sorted
    lhs_pack = (num_feat_partitions > 1 and not use_bcast) and \
               ((lhs_target == TargetCode.SRC and is_sorted == 0) or \
                (lhs_target == TargetCode.DST and is_sorted == 1) or \
                lhs_target == TargetCode.EDGE)
    rhs_pack = (num_feat_partitions > 1 and not use_bcast) and \
               ((rhs_target == TargetCode.SRC and is_sorted == 0) or \
                (rhs_target == TargetCode.DST and is_sorted == 1) or \
                rhs_target == TargetCode.EDGE)
    # compute
    out, reshapes = _sddmm_compute((nnz,) + out_shp, binary_op, lhs, rhs, \
                                            idx_target(lhs_target), idx_target(rhs_target),
                                            num_feat_partitions, lhs_pack, rhs_pack)
    # schedule
    s = te.create_schedule(out.op)
    if target == 'cuda':
        # cuda schedule
        if binary_op == 'dot' and reduce_size >= 32:
            # if dot product, use tree reduction
            _sddmm_cuda_tree_reduce(s, out)
        else:
            _sddmm_cuda_general(s, out)
    elif target == 'llvm':
        if num_feat_partitions == 1:
            _sddmm_cpu_general(s, out)
        else:
            _sddmm_cpu_feat_partition(s, out, binary_op, reshapes, reduce_size, num_feat_partitions)
    # prepare input
    f_input = []
    if lhs_target == TargetCode.SRC or rhs_target == TargetCode.SRC:
        f_input.append(adj_row_indices)
    if lhs_target == TargetCode.DST or rhs_target == TargetCode.DST:
        f_input.append(adj_col_indices)
    # use_idx should only be set when array packing is applied
    # otherwise mapping should be done outside of tvm
    if use_idx:
        f_input.append(edge_mapping)
    f_name = '_'.join(str(x) for x in [
        'sddmm', binary_op, 
         lhs_target, rhs_target,
         indice_type, feat_type
         ])
    f_input += [lhs, rhs, out]
    # bind autobroadcast buffer
    lhs_buffer = tvm.tir.decl_buffer(lhs.shape, lhs.dtype, name='lhs_buf', buffer_type='auto_broadcast')
    rhs_buffer = tvm.tir.decl_buffer(rhs.shape, rhs.dtype, name='rhs_buf', buffer_type='auto_broadcast')
    binds = {}
    if use_bcast:
        binds = {lhs:lhs_buffer, rhs:rhs_buffer}
    # print(tvm.lower(s, f_input, binds=binds))
    return tvm.build(s, f_input, target=target, name=f_name, binds=binds)

if __name__ == '__main__':
    target = 'llvm'
    lhs_shp, rhs_shp = (8,), (8,)
    out_shp = (1,)
    nnz = 5
    num_rows = 10
    num_cols = 10
    indice_type = 'int32'
    feat_type = 'float32'
    f = sddmm('dot', nnz, num_rows, num_cols, 
         lhs_shp, rhs_shp, out_shp,
         indice_type, feat_type, use_idx=False,
         num_feat_partitions=2, target=target)
    # print(f.imported_modules[0].get_source())